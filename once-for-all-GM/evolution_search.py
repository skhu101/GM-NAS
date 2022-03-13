# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.
import argparse
import numpy as np
import os
import random
# import horovod.torch as hvd
import torch
import distributed
from elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from elastic_nn.networks import OFAMobileNetV3
from nas.accuracy_predictor.acc_dataset import AccuracyDataset

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1

import argparse
import numpy as np
import os
import random
# import horovod.torch as hvd
import torch
import distributed

# try:
#     import moxing as mox
# except Exception as e:
#     print("Can not find moxing.")

from elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3
from imagenet_codebase.run_manager import DistributedImageNetRunConfig
from imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager
from imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from imagenet_codebase.utils import download_url
from elastic_nn.training.progressive_shrinking import load_models
from nas.accuracy_predictor.arch_encoder import MobileNetArchEncoder
from evolution_ws import EvolutionFinder
import warnings
from torch import multiprocessing

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='normal', choices=[
    'normal', 'kernel', 'depth', 'expand',
])
parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
parser.add_argument("--local_rank", type=int)
parser.add_argument("--port", type=int, default=23346)
parser.add_argument('--initial_enc', type=str, default=None, help='initial enc path')
parser.add_argument('--initial_model', type=str, default=None, help='initial model path')
# parser.add_argument('--split_method', type=str, default='GM', choices=['GM', 'fewshot'], help='e.g. GM, fewshot')
# parser.add_argument('--submodel_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
args = parser.parse_args()

args.manual_seed = 0

args.lr_schedule_type = 'cosine'

args.base_batch_size = 40000
# args.base_batch_size = 128
args.valid_size = 10000

args.opt_type = 'sgd'
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.no_decay_keys = 'bn#bias'
args.fp16_allreduce = False

args.model_init = 'he_fout'
args.validation_frequency = 1
args.print_frequency = 10

args.n_worker = 8
args.resize_scale = 0.08
args.distort_color = 'tf'
# args.image_size = '128,160,192,224'
args.image_size = '224'
args.continuous_size = True
args.not_sync_distributed_image_size = False

# args.bn_momentum = 0.1
# args.bn_eps = 1e-5
# args.dropout = 0.1
# args.base_stage_width = 'proxyless'

args.width_mult_list = '1.0'
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_type = 'ce'

os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
os.environ.setdefault('run_name', 'default')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


if __name__ == '__main__':
    args.path = 'exp/acc_dataset'
    os.makedirs(args.path, exist_ok=True)
    # Initialize Horovod
    # hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    # num_gpus = hvd.size()
    rank, num_gpus = distributed.dist_init_pytorch(args.port, 'nccl', args.local_rank)
    # torch.cuda.set_device(rank)
    print("rank", distributed.get_rank())
    print("world_size", num_gpus)
    # if args.task != 'normal':
    #     args.teacher_path = "exp/normal/checkpoint/model_best.pth.tar"
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=rank)

    # print run config information
    if rank == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    args.ks_list = '3,5,7'
    args.expand_list = '3,4,6'
    args.depth_list = '2,3,4'

    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [int(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]

    net = OFAMobileNetV3(
        dropout_rate=0, width_mult_list=1.0, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
    )

    init = torch.load(args.initial_model, map_location='cpu')['state_dict']
    kernel_size_enc = torch.from_numpy(np.loadtxt(args.initial_enc, dtype=int))


    dynamic_batch_size = 1

    distributed_run_manager = DistributedRunManager(
        args.path, net, run_config, None, backward_steps=dynamic_batch_size, is_root=(rank == 0)
    )
    distributed_run_manager.save_config()

    distributed_run_manager.net.load_state_dict(init)
    print('load model')

    print("loaded kernel size enc: ", kernel_size_enc)

    arch_manager = MobileNetArchEncoder(kernel_size_enc=kernel_size_enc)

    evolution = EvolutionFinder(distributed_run_manager, arch_manager)

    best_valids, best_info = evolution.run_evolution_search(500, 600, input_size=224)
    print(best_valids, best_info)
