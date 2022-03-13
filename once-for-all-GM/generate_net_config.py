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
from elastic_nn.networks.ofa_mbv3 import MobileNetV3_layer_settting, OFAMobileNetV3_layer_settting

DynamicSeparableConv2d.KERNEL_TRANSFORM_MODE = 1

import argparse
import numpy as np
import os
import random
# import horovod.torch as hvd
import torch
import distributed
import json

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
from utils import make_divisible

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
parser.add_argument('--submodel_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()

def count_conv_flop(out_size, in_channels, out_channels, kernel_size, groups):
    out_h = out_w = out_size
    delta_ops = in_channels * out_channels * kernel_size * kernel_size * out_h * out_w / groups
    return delta_ops

def count_flops_given_config(net_config, image_size=224):
    flops = 0
    # first conv
    flops += count_conv_flop((image_size + 1) // 2, 3, net_config['first_conv']['out_channels'], 3, 1)
    # blocks
    fsize = (image_size + 1) // 2
    count = 0
    for block in net_config['blocks']:
        mb_conv = block['mobile_inverted_conv'] if 'mobile_inverted_conv' in block else block['conv']
        if mb_conv is None:
            continue
        out_fz = int((fsize - 1) / mb_conv['stride'] + 1)
        # if mb_conv['mid_channels'] is None:
        if 'in_channel_list' in mb_conv.keys():
            mb_conv['in_channels'] = mb_conv['in_channel_list'][0]
        if 'out_channel_list' in mb_conv.keys():
            mb_conv['out_channels'] = mb_conv['out_channel_list'][0]
        if 'kernel_size_list' in mb_conv.keys():
            mb_conv['kernel_size'] = mb_conv['kernel_size_list'][0]
        if 'expand_ratio_list' in mb_conv.keys():
            mb_conv['expand_ratio'] = mb_conv['expand_ratio_list'][0]
        mb_conv['mid_channels'] = round(mb_conv['in_channels'] * mb_conv['expand_ratio'])
        if mb_conv['expand_ratio'] != 1:
            # inverted bottleneck
            flops += count_conv_flop(fsize, mb_conv['in_channels'], mb_conv['mid_channels'], 1, 1)
        # depth conv
        flops += count_conv_flop(out_fz, mb_conv['mid_channels'], mb_conv['mid_channels'],
                                 mb_conv['kernel_size'], mb_conv['mid_channels'])
        if mb_conv['use_se']:
            # SE layer
            se_mid = make_divisible(mb_conv['mid_channels'] // 4, divisor=8)
            flops += count_conv_flop(1, mb_conv['mid_channels'], se_mid, 1, 1)
            flops += count_conv_flop(1, se_mid, mb_conv['mid_channels'], 1, 1)
        # point linear
        flops += count_conv_flop(out_fz, mb_conv['mid_channels'], mb_conv['out_channels'], 1, 1)
        fsize = out_fz
        count += 1
    # final expand layer
    flops += count_conv_flop(fsize, net_config['final_expand_layer']['in_channels'],
                             net_config['final_expand_layer']['out_channels'], 1, 1)
    # feature mix layer
    flops += count_conv_flop(1, net_config['feature_mix_layer']['in_channels'],
                             net_config['feature_mix_layer']['out_channels'], 1, 1)
    # classifier
    flops += count_conv_flop(1, net_config['classifier']['in_features'],
                             net_config['classifier']['out_features'], 1, 1)
    return flops / 1e6  # MFLOPs



if __name__ == '__main__':
    sample = {}

    # GM_split_3_edge_2_group_subnetwork_1 80.38 584
    sample['ks'] = [7, 7, 5, 5, 5, 5, 7, 3, 7, 3, 7, 7, 7, 7, 3, 3, 5, 3, 7, 5]
    sample['e'] = [3, 3, 4, 4, 4, 3, 6, 4, 6, 4, 3, 6, 6, 6, 6, 4, 6, 6, 6, 3]
    sample['d'] = [4, 4, 4, 4, 4]


    net = MobileNetV3_layer_settting(
        dropout_rate=0, width_mult_list=1.0, ks_list=sample['ks'], expand_ratio_list=sample['e'],
        depth_list=sample['d'],
    )
    flops = count_flops_given_config(net.config, image_size=224)
    print(flops)


    net_save_path = 'GM_split_3_edge_2_group_subnetwork_1/GM_split_3_edge_2_group_subnetwork_1.config'
    json.dump(net.config, open(net_save_path, 'w'), indent=4)
    print('Network configs dump to %s' % net_save_path)

