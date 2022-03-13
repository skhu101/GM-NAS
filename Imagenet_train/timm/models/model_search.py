import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import *
import random



class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in OPS_POOL:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
              op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class FixedOp(nn.Module):

    def __init__(self, C, stride, fixed_op):
        super(FixedOp, self).__init__()
        self.op = OPS[OPS_POOL[fixed_op]](C, stride, False)

        if 'pool' in OPS_POOL[fixed_op]:
            self.op = nn.Sequential(self.op, nn.BatchNorm2d(C, affine=False))


    def forward(self, x, weights):
        return self.op(x)


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, fixed_op=None):
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        self._bns = nn.ModuleList()

        if fixed_op is None:

            for i in range(self._steps):
                for j in range(2 + i):
                    stride = 2 if reduction and j < 2 else 1
                    op = MixedOp(C, stride)
                    self._ops.append(op)

        else:
            for i in range(self._steps):
                for j in range(2 + i):
                    if i == 0 and j == 0:
                        stride = 2 if reduction and j < 2 else 1
                        op = FixedOp(C, stride, fixed_op=fixed_op)
                        self._ops.append(op)
                    else:
                        stride = 2 if reduction and j < 2 else 1
                        op = MixedOp(C, stride)
                        self._ops.append(op)


    def forward(self, s0, s1, weights):
          s0 = self.preprocess0(s0)
          s1 = self.preprocess1(s1)

          states = [s0, s1]
          offset = 0
          for i in range(self._steps):
              s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
              offset += len(states)
              states.append(s)

          return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):

    def __init__(self, C, num_classes, layers=10, steps=6, multiplier=4, stem_multiplier=3, fixed_op=None):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier
        self.fixed_op = fixed_op

        C_curr = stem_multiplier*C
        self.stem = nn.Sequential(
          nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
          nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            # if i in [layers//3, 2*layers//3]:
            if i in [layers//3]:
              C_curr *= 2
              reduction = True
            else:
              reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, fixed_op=fixed_op)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier*C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)



    def random_mask(self):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))

        weights = [[0.0] * len(OPS_POOL) for i in range(k)]

        for i in range(len(weights)):
            weights[i][random.randint(0, len(OPS_POOL) - 1)] = 1.0



        # print(weights)
        return weights

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            weights = self.random_mask()
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits







