import os
import torch
import shutil
import time
import logging
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
# import .xnas_parameters as params

from collections import namedtuple

Dataset = namedtuple('Dataset',
                     ['num_classes',
                      'num_channels',
                      'hw',
                      'mean',
                      'std',
                      'initial_channels_factor',
                      'is_ImageFolder',
                      'def_resize'])

datasets = {'CIFAR10':
              Dataset(10, 3, [32, 32],
                      [0.49139968, 0.48215827, 0.44653124],
                      [0.24703233, 0.24348505, 0.26158768],
                      1,
                      False,
                      None),

            'ImageNet':
              Dataset(1000, 3, [224, 224],
                      [0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225],
                      1,
                      True,
                      None),
            }


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


# def infer(test_queue, model, report_freq=50):
#     top1 = AvgrageMeter()
#     top5 = AvgrageMeter()
#     model.eval()
#     samples = 0
#     infer_time = 0
#
#     for step, (input, target) in enumerate(test_queue):
#         input = Variable(input, requires_grad=False).cuda()
#         target = Variable(target, requires_grad=False).cuda(async=True)
#
#         ts = time.time()
#         logits = model(input)
#         te = time.time()
#         infer_time += (te - ts)
#
#         prec1, prec5 = accuracy(logits, target, topk=(1, 5))
#         n = input.size(0)
#         top1.update(prec1.data.item(), n)
#         top5.update(prec5.data.item(), n)
#
#         samples += n
#
#         if step % report_freq == 0:
#             logging.info('test %03d %f %f', step, top1.avg, top5.avg)
#
#     infer_time = infer_time / samples
#
#     return top1.avg, infer_time

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def data_transforms_cifar10():
    dataset = datasets['CIFAR10']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset.mean, dataset.std),
    ])

    return transform

def data_transforms_imagenet_valid():
    dataset = datasets['ImageNet']
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(dataset.mean, dataset.std),
    ])

    return transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    checkpoint = torch.load(model_path)
    if checkpoint.__contains__('model'):
        model.load_state_dict(checkpoint['model'], strict=False)
    elif checkpoint.__contains__('state_dict'):
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


#  For uniform_end mode: reduction_indices = num_layers - place_reduction_cells(...)
def place_reduction_cells(num_layers, num_reductions, mode='original', bias=0,
                          verbose=False):
    if mode == 'original':
        reduction_indices = np.array(
            num_layers * np.arange(1, num_reductions + 1) // (num_reductions + 1))
    elif mode == 'uniform_start':  # When cant be fully-uniform, bias towards start/end of net.
        normal_len = (num_layers - num_reductions) / (
            num_reductions + 1)  # (real) number of normal cells
        #  between consecutive reduction cells.
        if normal_len == int(normal_len):
            reduction_indices = bias + (
                (int(normal_len) + 1) * np.arange(1, num_reductions + 1) - 1)
        else:
            # print("num_layers: ", num_layers, "num_reductions: ", num_reductions, "normal_len: ", normal_len, "bias: ", bias)
            if num_reductions == 1:
                # print("num_reductions == 1")
                reduction_indices = bias + int(normal_len)
            else:
                reduction_indices = np.concatenate((int(normal_len + bias),
                                                    place_reduction_cells(
                                                        num_layers - int(normal_len + 1),
                                                        num_reductions - 1, mode,
                                                        bias + int(normal_len) + 1)),
                                                   axis=None)
    else:
        assert False, "No such mode."
    if verbose:
        print("Network's reduction cell indices: ", reduction_indices)
    return reduction_indices


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)