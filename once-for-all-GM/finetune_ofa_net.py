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
import json

try:
    import moxing as mox
except Exception as e:
    pass

from elastic_nn.modules.dynamic_op import DynamicSeparableConv2d
from elastic_nn.networks.ofa_mbv3 import OFAMobileNetV3
from imagenet_codebase.run_manager import DistributedImageNetRunConfig
from imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager
from imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from imagenet_codebase.utils import download_url
from elastic_nn.training.progressive_shrinking import load_models
import warnings
from torch import multiprocessing

try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='finetune', choices=[
    'normal', 'kernel', 'depth', 'expand', 'finetune'
])
parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
parser.add_argument("--local_rank", type=int)
parser.add_argument('--path', type=str, default='exp/normal')
parser.add_argument('--subnet_settings_path', type=str, default='exp/subnetwork_acc_79.26_flops_300.txt')
# parser.add_argument('--submodel_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()
if args.task == 'normal':
    args.path = 'exp/normal'
    args.dynamic_batch_size = 1
    args.n_epochs = 180
    args.base_lr = 0.1625
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = '7'
    args.expand_list = '6'
    args.depth_list = '4'
    args.kd_ratio = .0
elif args.task == 'kernel':
    args.path = 'exp/normal2kernel'
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 0.06
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '6'
    args.depth_list = '4'
    args.kd_ratio = 1.0
elif args.task == 'depth':
    args.path = 'exp/kernel2kernel_depth/phase%d' % args.phase
    args.dynamic_batch_size = 2
    args.kd_ratio = 1.0
    if args.phase == 1:
        args.n_epochs = 20
        args.base_lr = 0.005
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = '3,5,7'
        args.expand_list = '6'
        args.depth_list = '3,4'
    else:
        args.n_epochs = 120
        args.base_lr = 0.015
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = '3,5,7'
        args.expand_list = '6'
        args.depth_list = '2,3,4'
elif args.task == 'expand':
    args.path = 'exp/kernel_depth2kernel_depth_width/phase%d' % args.phase
    args.dynamic_batch_size = 4
    args.kd_ratio = 1.0
    if args.phase == 1:
        args.n_epochs = 20
        args.base_lr = 0.005
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = '3,5,7'
        args.expand_list = '4,6'
        args.depth_list = '2,3,4'
    else:
        args.n_epochs = 120
        args.base_lr = 0.015
        args.warmup_epochs = 5
        args.warmup_lr = -1
        args.ks_list = '3,5,7'
        args.expand_list = '3,4,6'
        args.depth_list = '2,3,4'
elif args.task == 'finetune':
    args.dynamic_batch_size = 1
    args.kd_ratio = 1.0
    args.n_epochs = 75
    args.base_lr = 0.005
    args.warmup_epochs = 0
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '3,4,6'
    args.depth_list = '2,3,4'


args.manual_seed = 0

args.lr_schedule_type = 'cosine'

# args.base_batch_size = 128
# args.valid_size = 10000
args.base_batch_size = 16
args.valid_size = None

args.opt_type = 'sgd'
args.momentum = 0.9
args.no_nesterov = False
args.weight_decay = 3e-5
args.label_smoothing = 0.1
args.mixup_alpha = 0.2
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

args.bn_momentum = 0.1
args.bn_eps = 1e-5
args.dropout = 0.1
args.base_stage_width = 'proxyless'

args.width_mult_list = '1.0'
args.dy_conv_scaling_mode = 1
args.independent_distributed_sampling = False

args.kd_type = 'ce'

os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
os.environ.setdefault('run_name', 'default')
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

if __name__ == '__main__':
    os.makedirs(args.path, exist_ok=True)
    # Initialize Horovod
    # hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    # num_gpus = hvd.size()
    rank, num_gpus = distributed.dist_init_pytorch(23336, 'nccl', args.local_rank)
    # torch.cuda.set_device(rank)
    print("rank", distributed.get_rank())
    print("world_size", num_gpus)
    if args.task != 'normal':
        args.teacher_path = "exp/normal/checkpoint/model_best.pth.tar"
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    #
    # image size
    args.image_size = [int(img_size) for img_size in args.image_size.split(',')]
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]
    MyRandomResizedCrop.CONTINUOUS = args.continuous_size
    MyRandomResizedCrop.SYNC_DISTRIBUTED = not args.not_sync_distributed_image_size

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    args.init_lr = args.base_lr * num_gpus  # linearly rescale the learning rate
    if args.warmup_lr < 0:
        args.warmup_lr = args.base_lr
    args.train_batch_size = args.base_batch_size
    args.test_batch_size = args.base_batch_size * 4
    run_config = DistributedImageNetRunConfig(**args.__dict__, num_replicas=num_gpus, rank=rank)

    # print run config information
    if rank == 0:
        print('Run config:')
        for k, v in run_config.config.items():
            print('\t%s: %s' % (k, v))

    if args.dy_conv_scaling_mode == -1:
        args.dy_conv_scaling_mode = None
    DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = args.dy_conv_scaling_mode

    # build net from args
    args.width_mult_list = [float(width_mult) for width_mult in args.width_mult_list.split(',')]
    args.ks_list = [int(ks) for ks in args.ks_list.split(',')]
    args.expand_list = [int(e) for e in args.expand_list.split(',')]
    args.depth_list = [int(d) for d in args.depth_list.split(',')]

    net = OFAMobileNetV3(
        n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout, base_stage_width=args.base_stage_width, width_mult_list=args.width_mult_list,
        ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list
    )
    # teacher model
    if args.kd_ratio > 0:
        args.teacher_model = OFAMobileNetV3(
            n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
            dropout_rate=0, width_mult_list=1.0, ks_list=7, expand_ratio_list=6, depth_list=4,
        )
        args.teacher_model.cuda()

    """ Distributed RunManager """
    # Horovod: (optional) compression algorithm.
    # compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    # distributed_run_manager = DistributedRunManager(
    #     args.path, net, run_config, compression, backward_steps=args.dynamic_batch_size, is_root=(rank == 0)
    # )
    distributed_run_manager = DistributedRunManager(
        args.path, net, run_config, None, backward_steps=args.dynamic_batch_size, is_root=(rank == 0)
    )
    distributed_run_manager.save_config()
    # hvd broadcast
    # distributed_run_manager.broadcast()

    # load ofa net
    init = torch.load('exp/kernel_depth2kernel_depth_width/phase2/checkpoint/model_best.pth.tar', map_location='cpu')[
        'state_dict']
    distributed_run_manager.net.load_state_dict(init)

    # load teacher net weights
    if args.kd_ratio:
        load_models(distributed_run_manager, args.teacher_model, model_path=args.teacher_path)

    # reading the model_config from the file
    with open(args.subnet_settings_path) as f:
        model_config = f.read()
    subnet_settings = json.loads(model_config)
    # print(subnet_settings)
    distributed_run_manager.net.set_active_subnet(ks=subnet_settings['ks'], e=subnet_settings['e'], d=subnet_settings['d'])
    # kernel_size_enc = torch.from_numpy(
    #     np.loadtxt("exp/subnetwork_" + str(args.submodel_id) + '_kernel_size_enc.txt', dtype=int))
    # for i in range(1, len(distributed_run_manager.net.blocks)):
    #     distributed_run_manager.net.set_encoding(i, kernel_size_enc[i - 1])
    # print("loaded kernel size enc")

    # if rank == 0:
    #     print('\n' + 'sub_model id ' + str(args.submodel_id) + ' enc: ')
    #     for i in range(1, len(distributed_run_manager.net.blocks)):
    #         print('\n' + str(distributed_run_manager.net.blocks[i].mobile_inverted_conv.kernel_size_enc))

    # training
    from elastic_nn.training.progressive_shrinking import finetune_net, finetune_validate

    validate_func_dict = {'image_size_list': {224} if isinstance(args.image_size, int) else sorted({160, 224}),
                          'width_mult_list': sorted({0, len(args.width_mult_list) - 1}),
                          'ks_list': sorted({min(args.ks_list), max(args.ks_list)}),
                          'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
                          'depth_list': sorted({min(net.depth_list), max(net.depth_list)})}


    distributed_run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
                                      finetune_validate(distributed_run_manager, subnet_settings=subnet_settings, **validate_func_dict), 'valid')

    finetune_net(distributed_run_manager, args, subnet_settings,
          lambda _run_manager, epoch, is_test, subnet_settings: finetune_validate(_run_manager, epoch, is_test, subnet_settings=subnet_settings, **validate_func_dict))


    # if args.task == 'normal':
    #     validate_func_dict['ks_list'] = sorted(args.ks_list)
    #     if distributed_run_manager.start_epoch == 0:
    #         distributed_run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
    #                                           validate(distributed_run_manager, **validate_func_dict), 'valid')
    #     train(distributed_run_manager, args,
    #           lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
    # elif args.task == 'kernel':
    #     validate_func_dict['ks_list'] = sorted(args.ks_list)
    #     if distributed_run_manager.start_epoch == 0:
    #         print("doing kernel")
    #         load_models(distributed_run_manager, distributed_run_manager.net,
    #                     model_path='exp/normal/checkpoint/model_best.pth.tar')
    #         print("loaded model")
    #         distributed_run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
    #                                           validate(distributed_run_manager, **validate_func_dict), 'valid')
    #     train(distributed_run_manager, args,
    #           lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
    # elif args.task == 'depth':
    #     from elastic_nn.training.progressive_shrinking import supporting_elastic_depth
    #
    #     supporting_elastic_depth(train, distributed_run_manager, args, validate_func_dict)
    # elif args.task == 'expand':
    #     from elastic_nn.training.progressive_shrinking import supporting_elastic_expand
    #
    #     supporting_elastic_expand(train, distributed_run_manager, args, validate_func_dict)

