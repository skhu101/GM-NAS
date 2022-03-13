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

try:
    import moxing as mox
except Exception as e:
    pass
    # print("Can not find moxing.")

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
parser.add_argument('--task', type=str, default='normal', choices=[
    'normal', 'kernel', 'depth', 'expand',
])
parser.add_argument('--phase', type=int, default=1, choices=[1, 2])
parser.add_argument("--local_rank", type=int)

#### weight-sharing NAS
parser.add_argument('--dis_metric', type=str, default='cos', choices=['per-filter-cos', 'cos', 'mse'])
parser.add_argument('--split_eid', type=int, default=1, choices=[1,2,3,4,5,6,7,8,9,10,11,12,13], help='for checking gradient only')
parser.add_argument('--split_ckpts', type=str, default=None, help='e.g. 20,40,60')
parser.add_argument('--skip_final_split', type=int, default=0, help='whether to split at split_ckpts[-1]; used for reproducing few-shot NAS only')
parser.add_argument('--split_crit', type=str, default='grad', choices=['grad', 'fewshot'], help='e.g. grad, fewshot')
parser.add_argument('--edge_crit', type=str, default='rand', choices=['grad', 'rand'], help='e.g. grad,rand')
parser.add_argument('--split_num', type=str, default=None, help='split into how many groups?')
parser.add_argument('--select_edge_1', help='select edge 1 for splitting', action='store_true')
parser.add_argument('--exp_name', type=str, default='exp', help='experiment directory')
parser.add_argument('--initial_enc', type=str, default=None, help='initial enc path')


args = parser.parse_args()

args.split_ckpts = [int(x) for x in args.split_ckpts.split(',')]
args.split_num = [int(x) for x in args.split_num.split(',')]

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
    args.path = args.exp_name+'/normal2kernel'
    args.dynamic_batch_size = 1
    args.n_epochs = 120
    args.base_lr = 0.06
    args.warmup_epochs = 5
    args.warmup_lr = -1
    args.ks_list = '3,5,7'
    args.expand_list = '6'
    args.depth_list = '4'
    args.kd_ratio = 1.0

args.manual_seed = 0

args.lr_schedule_type = 'cosine'

args.base_batch_size = 128
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
args.image_size = '128,160,192,224'
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


def get_trained_result(from_loc, to_loc):
    mox.file.copy_parallel(from_loc, to_loc)
    print("file copied")


if __name__ == '__main__':
    os.makedirs(args.path, exist_ok=True)
    # Initialize Horovod
    # hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    # num_gpus = hvd.size()
    rank, num_gpus = distributed.dist_init_pytorch(23333, 'nccl', args.local_rank)
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

    # net = OFAMobileNetV3(
    #     n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
    #     dropout_rate=args.dropout, base_stage_width=args.base_stage_width, width_mult_list=args.width_mult_list,
    #     ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list
    # )
    get_new_model  = lambda: OFAMobileNetV3(
        n_classes=run_config.data_provider.n_classes, bn_param=(args.bn_momentum, args.bn_eps),
        dropout_rate=args.dropout, base_stage_width=args.base_stage_width, width_mult_list=args.width_mult_list,
        ks_list=args.ks_list, expand_ratio_list=args.expand_list, depth_list=args.depth_list
    )
    net = get_new_model()


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
    # distributed_run_manager = DistributedRunManager(
    #     args.path, net, run_config, None, backward_steps=args.dynamic_batch_size, is_root=(rank == 0)
    # )
    get_new_distributed_run_manager = lambda net: DistributedRunManager(
        args.path, net, run_config, None, backward_steps=args.dynamic_batch_size, is_root=(rank == 0)
    )
    distributed_run_manager = get_new_distributed_run_manager(net)
    distributed_run_manager.save_config()
    # hvd broadcast
    # distributed_run_manager.broadcast()

    # load teacher net weights
    if args.kd_ratio:
        load_models(distributed_run_manager, args.teacher_model, model_path=args.teacher_path)

    # training
    from elastic_nn.training.progressive_shrinking import validate, train, validate_supernet, train_supernet, split_supernet

    validate_func_dict = {'image_size_list': {224} if isinstance(args.image_size, int) else sorted({160, 224}),
                          'width_mult_list': sorted({0, len(args.width_mult_list) - 1}),
                          'ks_list': sorted({min(args.ks_list), max(args.ks_list)}),
                          'expand_ratio_list': sorted({min(args.expand_list), max(args.expand_list)}),
                          'depth_list': sorted({min(net.depth_list), max(net.depth_list)})}
    if args.task == 'normal':
        validate_func_dict['ks_list'] = sorted(args.ks_list)
        if distributed_run_manager.start_epoch == 0:
            distributed_run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
                                              validate(distributed_run_manager, **validate_func_dict), 'valid')
        train(distributed_run_manager, args,
              lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict))
    elif args.task == 'kernel':

        if args.initial_enc is not None:

            kernel_size_enc = torch.from_numpy(np.loadtxt(args.initial_enc, dtype=int))
            # set initial encoding
            for id in range(1, len(net.blocks)):
                distributed_run_manager.net.blocks[id].mobile_inverted_conv.kernel_size_enc.copy_(kernel_size_enc[id-1])

            if rank == 0:
                for id in range(1, len(net.blocks)):
                    print(distributed_run_manager.net.blocks[id].mobile_inverted_conv.kernel_size_enc)

        if args.select_edge_1:
            split_eids = [1]
            split_eids.extend(np.random.permutation(range(2, 21)))
        else:
            split_eids = []
            split_eids.extend(np.random.permutation(range(1, 21)))
        supernets_run_manager = [[net, distributed_run_manager]]
        start_epoch = 0
        split_id = 0

        # need to comment out
        # args.validation_frequency = 1
        args.warmup_epochs = 0

        for index, end_epoch in enumerate(args.split_ckpts):
            new_supernets_run_manager = []
            count = 0
            split_num = args.split_num[index]

            ## train all supernets at current level, and split them along the way
            for cur_model, cur_distributed_run_manager in supernets_run_manager:
                cur_distributed_run_manager.start_epoch = start_epoch
                cur_distributed_run_manager.end_epoch = end_epoch

                # joint training
                # cur_arch_search_run_manager.train(fix_net_weights=args.debug)
                validate_func_dict['ks_list'] = sorted(args.ks_list)
                if cur_distributed_run_manager.start_epoch == 0:
                    print("doing kernel")
                    load_models(cur_distributed_run_manager, cur_distributed_run_manager.net,
                                model_path='exp/normal/checkpoint/model_best.pth.tar')
                    print("loaded model")
                    cur_distributed_run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
                                                      validate_supernet(cur_distributed_run_manager, **validate_func_dict), 'valid')
                train_supernet(cur_distributed_run_manager, args,
                      lambda _run_manager, epoch, is_test: validate_supernet(_run_manager, epoch, is_test, **validate_func_dict))

                if args.split_crit == 'grad':
                    if args.edge_crit == 'rand':
                        split_eid = split_eids[split_id]
                    elif args.edge_crit == 'grad':
                        split_eid = None
                elif args.split_crit == 'fewshot':
                    split_eid = args.split_eid

                if args.select_edge_1 and split_id == 0:
                    split_eid = split_eids[split_id]

                encs_splitted, select_eid = split_supernet(cur_distributed_run_manager, args, split_eid=split_eid, split_crit=args.split_crit, split_num=split_num, dis_metric=args.dis_metric)

                # spawn new supernets from the currently trained supernet
                for enc in encs_splitted:
                    # copy sub_architect and sub_scheduler
                    sub_model = get_new_model()

                    cur_distributed_run_manager.save_model({
                        'optimizer': cur_distributed_run_manager.optimizer.state_dict(),
                        'state_dict': cur_distributed_run_manager.net.state_dict(),
                    }, model_name='ckpt_' + str(end_epoch) + '_id_' + str(count) + '_checkpoint.pth.tar')

                    # copy encoding
                    for id in range(1, len(sub_model.blocks)):
                        sub_model.blocks[id].mobile_inverted_conv.kernel_size_enc.copy_(
                            cur_model.blocks[id].mobile_inverted_conv.kernel_size_enc)

                    sub_distributed_run_manager = get_new_distributed_run_manager(sub_model)
                    sub_distributed_run_manager.load_model(
                        model_fname=str(args.path) + '/checkpoint/ckpt_' + str(end_epoch) + '_id_' + str(
                            count) + '_checkpoint.pth.tar')

                    new_supernets_run_manager.append((sub_model, sub_distributed_run_manager))

                    sub_model.set_encoding(select_eid, enc)

                    if rank == 0:
                        print('\n' + 'sub_model id: ' + str(count))
                        kernel_size_enc_tensor = torch.zeros(len(sub_model.blocks)-1, 3)
                        for id in range(1, len(sub_model.blocks)):
                            print('\n' + str(sub_model.blocks[id].mobile_inverted_conv.kernel_size_enc))
                            kernel_size_enc_tensor[id-1] = sub_model.blocks[id].mobile_inverted_conv.kernel_size_enc

                        np.savetxt(str(args.exp_name) + '/subnetwork_' + str(count) + '_kernel_size_enc.txt', kernel_size_enc_tensor.numpy(), fmt='%i')
                    count += 1

                split_id += 1

                del cur_model, cur_distributed_run_manager

            ## move on to the next level
            start_epoch = end_epoch
            supernets_run_manager = new_supernets_run_manager

        ## train all supernets at current level, and split them along the way
        count = 0
        for cur_model, cur_arch_search_run_manager in supernets_run_manager:
            if rank == 0:
                print('\n' + 'sub_model id: ' + str(count))
            cur_arch_search_run_manager.start_epoch = start_epoch
            cur_arch_search_run_manager.end_epoch = cur_arch_search_run_manager.run_config.n_epochs

            cur_arch_search_run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' %
                                              validate(cur_arch_search_run_manager, **validate_func_dict), 'valid')
            # joint training
            train_supernet(distributed_run_manager, args,
                  lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict), cur_model_id=count)
            count += 1
