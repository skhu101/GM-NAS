import torch
import torch.nn as nn
import math
import numbers

from torch.nn.quantized.modules import FloatFunctional
from torch.autograd.function import Function
from .params_factory import unify_args, merge_unify_args, filter_kwargs, merge
from torch._six import container_abcs
from itertools import repeat
import torch.distributed as dist


OPS_POOL = ['none', 'max_pool_3x3', 'skip_connect', 'sep_conv_3x3', 'sep_conv_5x5', 'sep_conv_7x7']
            # ,'dil_conv_3x3', 'conv_7x1_1x7']


OPS = {
      'none' : lambda C, stride, affine: Zero(stride),
      'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
      'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
      'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
      'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
      'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
      'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
      'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
      'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
        ),

      'ir_k3': lambda C, stride, affine: IRFBlock(
        C, C, stride=stride, kernel_size=3),

      'ir_k5': lambda C, stride, affine: IRFBlock(
        C, C, stride=stride, kernel_size=5),
}

class ReLUConvBN(nn.Module):

      def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
          super(ReLUConvBN, self).__init__()
          self.op = nn.Sequential(
              nn.ReLU(inplace=False),
              nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
              nn.BatchNorm2d(C_out, affine=affine)
            )

      def forward(self, x):
            return self.op(x)

class DilConv(nn.Module):
    
      def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
          super(DilConv, self).__init__()
          self.op = nn.Sequential(
              nn.ReLU(inplace=False),
              nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
              nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
              nn.BatchNorm2d(C_out, affine=affine),
              )

      def forward(self, x):
          return self.op(x)


class SepConv(nn.Module):
    
      def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
          super(SepConv, self).__init__()
          self.op = nn.Sequential(
              nn.ReLU(inplace=False),
              nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
              nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
              nn.BatchNorm2d(C_in, affine=affine),
              nn.ReLU(inplace=False),
              nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
              nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
              nn.BatchNorm2d(C_out, affine=affine),
              )

      def forward(self, x):
          return self.op(x)


class Identity(nn.Module):

      def __init__(self):
          super(Identity, self).__init__()

      def forward(self, x):
          return x


class Zero(nn.Module):

      def __init__(self, stride):
          super(Zero, self).__init__()
          self.stride = stride

      def forward(self, x):
          if self.stride == 1:
            return x.mul(0.)
          return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

      def __init__(self, C_in, C_out, affine=True):
          super(FactorizedReduce, self).__init__()
          assert C_out % 2 == 0
          self.relu = nn.ReLU(inplace=False)
          self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
          self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
          self.bn = nn.BatchNorm2d(C_out, affine=affine)

      def forward(self, x):
          x = self.relu(x)
          out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
          out = self.bn(out)
          return out






################################################ Belows are mobile operations ###############################################





class TorchAdd(nn.Module):
    """Wrapper around torch.add so that all ops can be found at build"""

    def __init__(self):
        super().__init__()
        self.add_func = FloatFunctional()

    def forward(self, x, y):
        return self.add_func.add(x, y)

class TorchAddScalar(nn.Module):
    """ Wrapper around torch.add so that all ops can be found at build
        y must be a scalar, needed for quantization
    """

    def __init__(self):
        super().__init__()
        self.add_func = FloatFunctional()

    def forward(self, x, y):
        return self.add_func.add_scalar(x, y)


class TorchMultiply(nn.Module):
    """Wrapper around torch.mul so that all ops can be found at build"""

    def __init__(self):
        super().__init__()
        self.mul_func = FloatFunctional()

    def forward(self, x, y):
        return self.mul_func.mul(x, y)


class TorchMulScalar(nn.Module):
    """Wrapper around torch.mul so that all ops can be found at build
        y must be a scalar, needed for quantization
    """

    def __init__(self):
        super().__init__()
        self.mul_func = FloatFunctional()

    def forward(self, x, y):
        return self.mul_func.mul_scalar(x, y)



class AddWithDropConnect(nn.Module):
    """ Apply drop connect on x before adding with y """

    def __init__(self, drop_connect_rate):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        self.add = TorchAdd()

    def drop_connect_batch(self, inputs, drop_prob, training):
        """ Randomly drop batch during training """
        assert drop_prob < 1.0
        "Invalid drop_prob {drop_prob}"
        if not training or drop_prob == 0.0:
            return inputs
        batch_size = inputs.shape[0]
        keep_prob = 1 - drop_prob
        random_tensor = (
            torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
            + keep_prob
        )
        binary_tensor = torch.floor(random_tensor)
        output = inputs / keep_prob * binary_tensor
        return output

    def forward(self, x, y):
        xx = self.drop_connect_batch(x, self.drop_connect_rate, self.training)
        return self.add(xx, y)


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class Upsample(nn.Module):
    def __init__(
        self, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
        super(Upsample, self).__init__()
        self.size = size
        self.scale = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def ntuple(self, n):
        def parse(x):
            if isinstance(x, container_abcs.Iterable):
                return x
            return tuple(repeat(x, n))

        return parse


    def interpolate(
            self, input, size=None, scale_factor=None, mode="nearest", align_corners=None):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        def _check_size_scale_factor(dim):
            if size is None and scale_factor is None:
                raise ValueError("either size or scale_factor should be defined")
            if size is not None and scale_factor is not None:
                raise ValueError("only one of size or scale_factor should be defined")
            if (
                                scale_factor is not None
                        and isinstance(scale_factor, tuple)
                    and len(scale_factor) != dim
            ):
                raise ValueError(
                    "scale_factor shape must match input shape. "
                    "Input is {}D, scale_factor size is {}".format(dim, len(scale_factor))
                )

        def _output_size(dim):
            _check_size_scale_factor(dim)
            if size is not None:
                return size
            scale_factors = self.ntuple(dim)(scale_factor)
            # math.floor might return float in py2.7
            return [
                int(math.floor(input.size(i + 2) * scale_factors[i])) for i in range(dim)
                ]

        output_shape = tuple(_output_size(2))
        output_shape = input.shape[:-2] + output_shape
        return _NewEmptyTensorOp.apply(input, output_shape)



    def forward(self, x):
        return self.interpolate(x, size=self.size, scale_factor=self.scale, mode=self.mode,
                                align_corners=self.align_corners)

class GroupNorm(torch.nn.GroupNorm):
    def forward(self, x):
        if x.numel() > 0:
            return super(GroupNorm, self).forward(x)

        # get output shape
        output_shape = x.shape
        return _NewEmptyTensorOp.apply(x, output_shape)

class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


class NaiveSyncBatchNorm(nn.BatchNorm2d):
    """
    torch.nn.SyncBatchNorm has bugs. Use this before it is fixed.
    """

    def forward(self, input):
        if get_world_size() == 1 or not self.training:
            return super().forward(input)

        assert input.shape[0] > 0, "SyncBatchNorm does not support empty inputs"
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig = nn.Sigmoid()
        self.mul = TorchMultiply()

    def forward(self, x):
        return self.mul(x, self.sig(x))

class HSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU6(inplace=True)
        self.add_scalar = TorchAddScalar()
        self.mul_scalar = TorchMulScalar()

    def forward(self, x):
        # return self.relu(x + 3.0) / 6.0
        return self.mul_scalar(self.relu(self.add_scalar(x, 3.0)), 1.0 / 6.0)


class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        return nn.functional.channel_shuffle(x, self.groups)


def build_upsample_neg_stride(name=None, stride=None, **kwargs):
    """ Use negative stride to represent scales, i.e., stride=-2 means scale=2
        Return upsample op if the stride is negative, return None otherwise
        Reset and return the stride to 1 if it is negative
    """
    if name is None:
        return None, stride

    if isinstance(stride, numbers.Number):
        if stride > 0:
            return None, stride
        stride = (stride, stride)
    assert isinstance(stride, (tuple, list))

    neg_strides = all(x < 0 for x in stride)
    if not neg_strides:
        return None, stride

    scales = [-x for x in stride]
    if name == "default":
        ret = Upsample(scale_factor=scales, **kwargs)
    else:
        ret = None
    return ret, 1



class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, conv_args=None, bn_args=None, relu_args='relu',
                 upsample_args='default',
                 # additional arguments for conv
                 **kwargs):
        super().__init__()

        if conv_args is None:
            conv_args = 'conv'

        if bn_args is None:
            bn_args = 'bn'

        conv_full_args = merge_unify_args(conv_args, kwargs)
        conv_stride = conv_full_args.pop("stride", 1)
        # build upsample op if stride is negative


        upsample_op, conv_stride = self.build_upsample_neg_stride(
            stride=conv_stride, **unify_args(upsample_args)
        )
        # build conv
        conv_op = self.build_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=conv_stride,
            **conv_full_args,
        )

        # register in order
        self.conv = conv_op

        self.bn = ( self.build_bn(num_channels=out_channels, **unify_args(bn_args))
            if bn_args is not None
            else None
        )
        self.relu = (
            self.build_relu(num_channels=out_channels, **unify_args(relu_args))
            if relu_args is not None
            else None
        )
        self.upsample = upsample_op

        self.out_channels = out_channels


    def build_upsample_neg_stride(self, name=None, stride=None, **kwargs):
        """ Use negative stride to represent scales, i.e., stride=-2 means scale=2
            Return upsample op if the stride is negative, return None otherwise
            Reset and return the stride to 1 if it is negative
        """
        if name is None:
            return None, stride

        if isinstance(stride, numbers.Number):
            if stride > 0:
                return None, stride
            stride = (stride, stride)
        assert isinstance(stride, (tuple, list))

        neg_strides = all(x < 0 for x in stride)
        if not neg_strides:
            return None, stride

        scales = [-x for x in stride]
        if name == "default":
            ret = Upsample(scale_factor=scales, **kwargs)
        else:
            ret = None
        return ret, 1

    def build_conv(self, name="conv", in_channels=None, out_channels=None, weight_init="kaiming_normal",
            **conv_args):

        def _init_conv_weight(op, weight_init="kaiming_normal"):
            assert weight_init in [None, "kaiming_normal"]
            if weight_init is None:
                return
            if weight_init == "kaiming_normal":
                nn.init.kaiming_normal_(op.weight, mode="fan_out", nonlinearity="relu")
                if hasattr(op, "bias") and op.bias is not None:
                    nn.init.constant_(op.bias, 0.0)

        if name is None or name == "none":
            return None
        if name == "conv":
            conv_args = filter_kwargs(nn.Conv2d, conv_args)
            if "kernel_size" not in conv_args:
                conv_args["kernel_size"] = 1
            ret = nn.Conv2d(in_channels, out_channels, **conv_args)
            _init_conv_weight(ret, weight_init)
            return ret
        if name == "conv3d":
            conv_args = filter_kwargs(nn.Conv3d, conv_args)
            if "kernel_size" not in conv_args:
                conv_args["kernel_size"] = 1
            ret = nn.Conv3d(in_channels, out_channels, **conv_args)
            _init_conv_weight(ret, weight_init)
            return ret
        if name == "linear":
            ret = nn.Linear(in_channels, out_channels, bias=True)
            return ret

    def build_bn(self, name, num_channels, zero_gamma=None, **bn_args):
        if name is None or name == "none":
            bn_op = None
        elif name == "bn":
            bn_op = nn.BatchNorm2d(num_channels, **bn_args)
            if zero_gamma is True:
                nn.init.constant_(bn_op.weight, 0.0)
        elif name == "sync_bn":
            bn_op = NaiveSyncBatchNorm(num_channels, **bn_args)
            if zero_gamma is True:
                nn.init.constant_(bn_op.weight, 0.0)
        elif name == "sync_bn_torch":
            bn_op = nn.SyncBatchNorm(num_channels, **bn_args)
            if zero_gamma is True:
                nn.init.constant_(bn_op.weight, 0.0)
        elif name == "gn":
            bn_op = GroupNorm(num_channels=num_channels, **bn_args)
        elif name == "instance":
            bn_op = nn.InstanceNorm2d(num_channels, **bn_args)
        elif name == "bn3d":
            bn_op = nn.BatchNorm3d(num_channels, **bn_args)
            if zero_gamma is True:
                nn.init.constant_(bn_op.weight, 0.0)
        else:
            bn_op = None

        return bn_op

    def build_relu(self, name=None, num_channels=None, **kwargs):
        inplace = kwargs.get("inplace", True)
        if name is None or name == "none":
            return None
        if name == "relu":
            return nn.ReLU(inplace=inplace)
        if name == "relu6":
            return nn.ReLU6(inplace=inplace)
        if name == "leakyrelu":
            return nn.LeakyReLU(inplace=inplace)
        if name == "prelu":
            return nn.PReLU(num_parameters=num_channels, **kwargs)
        if name == "hswish":
            # HSwish = torch.nn.Hardswish()
            return nn.ReLU6(inplace=inplace)
            # return HSwish()
        if name == "swish":
            return Swish()
        if name == "sig":
            return nn.Sigmoid()
        if name == "hsig":
            return HSigmoid()

        return None


    def forward(self, x):
        # NOTE: `torch.autograd.Function` (used for empty batch) is not supported
        # for scripting now, so we skip it in scripting mode
        # We should remove empty batch function after empty batch is fully supported
        # by pytorch
        # See https://github.com/pytorch/pytorch/issues/22329
        if x.numel() > 0 or self.empty_input is None or torch.jit.is_scripting():
            if self.conv is not None:
                x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            if self.upsample is not None:
                x = self.upsample(x)
        else:
            x = self.empty_input(x)
            if self.upsample is not None:
                x = self.upsample(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels, mid_channels, fc=False, sigmoid_type="sigmoid", relu_args="relu"):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if not fc:
            conv1_relu = ConvBNRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bn_args=None,
                                    relu_args=relu_args)
            conv2 = nn.Conv2d(mid_channels, in_channels, 1, 1, 0)
        else:
            conv1_relu = ConvBNRelu(
                in_channels,
                mid_channels,
                conv_args="linear",
                bn_args=None,
                relu_args=relu_args,
            )
            conv2 = nn.Linear(mid_channels, in_channels, bias=True)

        if sigmoid_type == "sigmoid":
            sig = nn.Sigmoid()
        elif sigmoid_type == "hsigmoid":
            sig = HSigmoid()
        else:
            raise Exception("Incorrect sigmoid_type {sigmoid_type}")

        self.se = nn.Sequential(conv1_relu, conv2, sig)
        self.use_fc = fc
        self.mul = TorchMultiply()

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.avg_pool(x)
        if self.use_fc:
            y = y.view(n, c)
        y = self.se(y)
        if self.use_fc:
            y = y.view(n, c, 1, 1).expand_as(x)
        return self.mul(x, y)


def get_divisible_by(num, divisible_by, min_val=None):
    def py2_round(x):
        return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)

    ret = int(num)
    if min_val is None:
        min_val = divisible_by
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((py2_round(num / divisible_by) or 1) * divisible_by)
        if ret < 0.95 * num:
            ret += divisible_by
    return ret


def build_se(name=None, in_channels=None, mid_channels=None, width_divisor=None, **kwargs):
    if name is None:
        return None
    mid_channels = get_divisible_by(mid_channels, width_divisor)
    if name == "se":
        return SEModule(in_channels, mid_channels, **kwargs)
    if name == "se_fc":
        return SEModule(in_channels, mid_channels, fc=True, **kwargs)
    elif name == "se_hsig":
        return SEModule(in_channels, mid_channels, sigmoid_type="hsigmoid", **kwargs)
    raise Exception("Invalid SEModule arugments {name}")


class IRFBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        kernel_size=3,
        stride=1,
        bias=False,
        conv_args="conv",
        bn_args="bn",
        relu_args="relu",
        se_args=None,
        res_conn_args="default",
        upsample_args="default",
        width_divisor=8,
        pw_args=None,
        dw_args=None,
        pwl_args=None,
        dw_skip_bnrelu=False,
        pw_groups=1,
        dw_group_ratio=1,     # dw_group == mid_channels // dw_group_ratio
        pwl_groups=1,
        always_pw=False,
        less_se_channels=False,
        zero_last_bn_gamma=False,
        drop_connect_rate=None,
    ):
        super().__init__()

        mid_channels = get_divisible_by(
            in_channels * expansion, width_divisor
        )

        res_conn = self.build_residual_connect(
            name='default',
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            drop_connect_rate=drop_connect_rate
        )

        self.pw = None

        if in_channels != mid_channels or always_pw:
            self.pw = ConvBNRelu(
                in_channels=in_channels,
                out_channels=mid_channels,
                conv_args={
                    "kernel_size": 1,
                    "stride": 1,
                    "padding": 0,
                    "bias": bias,
                    "groups": pw_groups,
                    **merge_unify_args(conv_args, pw_args),
                },
                bn_args=bn_args,
                relu_args=relu_args,
            )

        self.shuffle = None

        if pw_groups > 1:
            self.shuffle = ChannelShuffle(pw_groups)
        # use negative stride for upsampling
        self.upsample, dw_stride = build_upsample_neg_stride(
            stride=stride, **unify_args(upsample_args)
        )

        dw_padding = (kernel_size // 2 if not (dw_args and "padding" in dw_args)
                      else dw_args.pop("padding"))

        self.dw = ConvBNRelu(
            in_channels=mid_channels,
            out_channels=mid_channels,
            conv_args={
                "kernel_size": kernel_size,
                "stride": dw_stride,
                "padding": dw_padding,
                "groups": mid_channels // dw_group_ratio,
                "bias": bias,
                **merge_unify_args(conv_args, dw_args),
            },
            bn_args=bn_args if not dw_skip_bnrelu else None,
            relu_args=relu_args if not dw_skip_bnrelu else None,
        )

        se_ratio = 0.25
        if less_se_channels:
            se_ratio /= expansion

        self.se = build_se(
            in_channels=mid_channels,
            mid_channels=int(mid_channels * se_ratio),
            width_divisor=width_divisor,
            **merge(relu_args=relu_args, kwargs=unify_args(se_args))
        )

        self.pwl = ConvBNRelu(
            in_channels=mid_channels,
            out_channels=out_channels,
            conv_args={
                "kernel_size": 1,
                "stride": 1,
                "padding": 0,
                "bias": bias,
                "groups": pwl_groups,
                **merge_unify_args(conv_args, pwl_args),
            },
            bn_args=bn_args,

            # {
            #     **bn_args,
            #     **{
            #         "zero_gamma": (
            #             zero_last_bn_gamma if res_conn is not None else False
            #         )
            #     },
            # },
            relu_args=None,
        )

        self.res_conn = res_conn
        self.out_channels = out_channels


    def build_residual_connect(self, name, in_channels, out_channels, stride, drop_connect_rate=None, **res_args):
        if name is None or name == "none":
            return None
        if name == "default":
            assert isinstance(stride, (numbers.Number, tuple, list))
            if isinstance(stride, (tuple, list)):
                stride_one = all(x == 1 for x in stride)
            else:
                stride_one = stride == 1
            if in_channels == out_channels and stride_one:
                if drop_connect_rate is None:
                    return TorchAdd()
                else:
                    return AddWithDropConnect(drop_connect_rate)
            else:
                return None

    def forward(self, x):
        y = x
        if self.pw is not None:
            y = self.pw(y)
        if self.shuffle is not None:
            y = self.shuffle(y)
        if self.upsample is not None:
            y = self.upsample(y)
        if self.dw is not None:
            y = self.dw(y)
        if self.se is not None:
            y = self.se(y)
        if self.pwl is not None:
            y = self.pwl(y)
        if self.res_conn is not None:
            y = self.res_conn(y, x)
        return y




























