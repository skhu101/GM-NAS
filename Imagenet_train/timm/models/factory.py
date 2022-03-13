from .registry import is_model, is_model_in_modules, model_entrypoint
from .helpers import load_checkpoint
from .layers import set_layer_config
from .model_search import Network
from .model import NetworkImageNet
from .model import NetworkCIFAR
from .flop import count_net_flops, count_net_params
from .ofa_model import OFAMobileNetV3
# from .xnas_model import NetworkImageNet

from .ofa_networks.proxyless_nets import ProxylessNASNets
from .ofa_networks.mobilenet_v3 import MobileNetV3
from .dnet_v3 import DNetV3
import json
import torch


import sys
sys.path.append("../..")


def create_model(
        model_name,
        pretrained=False,
        num_classes=1000,
        in_chans=3,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        num_classes (int): number of classes for final fully connected layer (default: 1000)
        in_chans (int): number of input channels / colors (default: 3)
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    # model_args = dict(pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)
    kwargs = dict(num_classes=num_classes, in_chans=in_chans)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # Parameters that aren't supported by all models should default to None in command line args,
    # remove them if they are present and not set so that non-supporting models don't break.
    if kwargs.get('drop_block_rate', None) is None:
        kwargs.pop('drop_block_rate', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    if kwargs.get('drop_path_rate', None) is None:
        kwargs.pop('drop_path_rate', None)

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # if source_name == 'hf_hub':
    #     # For model names specified in the form `hf_hub:path/architecture_name#revision`,
    #     # load model weights + default_cfg from Hugging Face hub.
    #     hf_default_cfg, model_name = load_model_config_from_hf(model_name)
    #     kwargs['external_default_cfg'] = hf_default_cfg  # FIXME revamp default_cfg interface someday

    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
            model = create_fn(pretrained=pretrained, **kwargs)
    else:
        with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
            if model_name == 'GM_ProxylessNAS_split_2_edge_2_3_group_subnetwork_1':
                net_config = json.load(
                    open("./timm/models/my_models/GM_ProxylessNAS_split_2_edge_2_3_group_subnetwork_1.config"))
                model = ProxylessNASNets.build_from_config(net_config)
                print('flop is ', count_net_flops(model, data_shape=(1, 3, 224, 224)))
            elif model_name == 'GM_OFA_split_2_edge_3_2_group_subnetwork_0_flops_587':
                net_config = json.load(open("./timm/models/my_models/GM_OFA_split_2_edge_3_2_group_subnetwork_0_flops_587.config"))
                model = MobileNetV3.build_from_config(net_config)
                print('flop is ', count_net_flops(model, data_shape=(1, 3, 224, 224)))
            elif model_name == 'GM_OFA_split_3_edge_2_group_subnetwork_1_flops_584':
                net_config = json.load(open("./timm/models/my_models/GM_OFA_split_3_edge_2_group_subnetwork_1_flops_584.config"))
                model = MobileNetV3.build_from_config(net_config)
                print('flop is ', count_net_flops(model, data_shape=(1, 3, 224, 224)))
            elif model_name == 'D-Net-big224':
                encoding = '23311-a02c12*32341-a02c12*02031-a02*031_64_211-2111-21111111111111111111111-211'
                model = DNetV3(encoding)
                print('flop is ', count_net_flops(model))
            else:
                raise RuntimeError('Unknown model (%s)' % model_name)

    print('param size is ', count_net_params(model))

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, use_ema=True)

    return model


