import os
import sys
sys.path.insert(0, '../')
from pprint import pformat
import numpy as np
import torch
import glob
import shutil
import nasbench201.utils as ig_utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from nasbench201.search_model_darts import TinyNetworkDarts
from nasbench201.search_model_darts_proj import TinyNetworkDartsProj
from nasbench201.search_model_ws import TinyNetworkWS
from nasbench201.cell_operations import SearchSpaceNames
from nasbench201.architect_ig import Architect
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from nas_201_api import NASBench201API as API
from nasbench201.projection import pt_project, mag_project, rsws_project

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet16-120'], help='choose dataset')
parser.add_argument('--search_space', type=str, default='nas-bench-201')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for alpha')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--group', type=str, default='none', help='experiment group')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
#### common
parser.add_argument('--fast', action='store_true', default=False, help='skip loading api which is slow')
parser.add_argument('--resume_epoch', type=int, default=0, help='0: from scratch; -1: resume from latest checkpoints')
parser.add_argument('--resume_expid', type=str, default='', help='e.g. search-darts-201-2')
parser.add_argument('--ckpt_interval', type=int, default=20, help='frequency for ckpting')
parser.add_argument('--expid_tag', type=str, default='none', help='extra tag for exp identification')
parser.add_argument('--log_tag', type=str, default='', help='extra tag for log during arch projection')
#### projection
parser.add_argument('--edge_decision', type=str, default='random', choices=['random'], help='which edge to be projected next')
parser.add_argument('--proj_crit', type=str, default='acc', choices=['loss', 'acc'], help='criteria for projection')
parser.add_argument('--proj_intv', type=int, default=5, help='fine tune epochs between two projections')
#### weight-sharing NAS
parser.add_argument('--method', type=str, default='ws', help='choose nas method')
parser.add_argument('--dis_metric', type=str, default='cos', choices=['per-filter-cos', 'cos', 'mse'])
parser.add_argument('--split_eids', type=str, default='none', help='use the specified eid order to split supernets')
parser.add_argument('--edge_crit', type=str, default='random-fully', choices=['predefine', 'random-perm', 'random-fully', 'grad'],
                                    help='criterion for determining which edge to split, "predefine": use provided split_eids')
parser.add_argument('--split_ckpts', type=str, default=None, help='e.g. 20,40,60')
parser.add_argument('--skip_final_split', type=int, default=0, help='whether to split at split_ckpts[-1]; used for reproducing few-shot NAS only')
parser.add_argument('--split_crit', type=str, default='grad', choices=['grad', 'fewshot'], help='e.g. 20,40,60')
parser.add_argument('--split_num', type=int, default=2, choices=[2,4], help='split into how many groups?')
parser.add_argument('--projection_warmup_epoch', type=int, default=0, help='how many epochs to train the supernet before final projection')
parser.add_argument('--supernet_train', type=str, default='darts', choices=['darts', 'pt', 'rsws'], help='arch selection method')
parser.add_argument('--projection', type=str, default='darts', choices=['darts', 'pt', 'rsws'], help='arch selection method')
#### eval
parser.add_argument('--mode', type=str, default="search", help='search vs rank, used to determine dataset split')
parser.add_argument('--ckpt_path', type=str, default='.')
parser.add_argument('--sids', type=str, default='0,1,2,3,4', help='supernet ids to be evaluated, parallel comp')
parser.add_argument('--bn_stats', type=int, default=0, help='track running stats')
args = parser.parse_args()


#### args augment
expid = args.save
if 'debug' in args.expid_tag: args.group = 'debug'
if 'dev'   in args.expid_tag: args.group = 'dev'
if args.supernet_train == 'rsws':
    args.weight_decay = 1e-4
args.fs_opt = False # dummy
args.save = os.path.join('../', args.ckpt_path.replace('/ckpts', ''))


#### dir management
## log file
if args.resume_epoch > 0:
    log_file = 'log_resume-{}_dev-{}_seed-{}_intv-{}'.format(args.resume_epoch, args.dev, args.seed, args.proj_intv)
    if args.log_tag != '': log_file += args.log_tag
else:
    log_file = f'log_eval_rank_[{args.sids}]'
if args.log_tag == 'debug': log_file = 'log_debug'
if args.mode != 'search': log_file += f'[rank]'
log_file += '.txt'
log_path = os.path.join(args.save, log_file)
if args.log_tag != 'debug' and os.path.exists(log_path):
    if input("WARNING: {} exists, override?[y/n]".format(log_file)) == 'y':
        print('proceed to override log file directory')
    else: exit(0)


#### logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
        
fh = logging.FileHandler(log_path, mode='w')
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(args.save + '/runs')

logging.info('======> log filename: %s', log_file)
logging.info(f'args = \n {pformat(args.__dict__)}')


#### macros
if args.dataset == 'cifar100':
    n_classes = 100
elif args.dataset == 'imagenet16-120':
    n_classes = 120
else:
    n_classes = 10


def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    gpu = ig_utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    # logging.info("args = %s", args)
    logging.info('gpu device = %d' % gpu)

    if not args.fast:
        api = API('../data/NAS-Bench-201-v1_0-e61699.pth')
    else:
        api = None


    #### model
    criterion = nn.CrossEntropyLoss()
    search_space = SearchSpaceNames[args.search_space]
    if args.method in ['darts', 'blank', 'darts-so']:
        helper = lambda: TinyNetworkDarts(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args)
    elif args.method in ['darts-proj', 'blank-proj']:
        helper = lambda: TinyNetworkDartsProj(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args)
    elif args.method in ['ws', 'ws-so']:
        helper = lambda: TinyNetworkWS(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args, track_running_stats=args.bn_stats)
    
    def get_new_model():
        model = helper()
        model.get_new_model = helper # for 2nd order
        return model
    
    model = get_new_model()
    model = model.cuda()
    logging.info("param size = %fMB", ig_utils.count_parameters_in_MB(model))

    get_new_architect = lambda model: Architect(model, args)
    architect = get_new_architect(model)


    #### data
    if args.dataset == 'cifar10':
        train_transform, valid_transform = ig_utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = ig_utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'imagenet16-120':
        import torchvision.transforms as transforms
        from nasbench201.DownsampledImageNet import ImageNet16
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22,  61.26, 65.09]]
        lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(16, padding=2), transforms.ToTensor(), transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        train_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=True, transform=train_transform, use_num_of_class_only=120)
        valid_data = ImageNet16(root=os.path.join(args.data,'imagenet16'), train=False, transform=train_transform, use_num_of_class_only=120)
        assert len(train_data) == 151700

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    if args.mode == 'rank':
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=0)
    else:
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True)
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True)
    
    
    #### resume (need some adjustment for nas-ws)
    for sid in args.sids.split(','):
        try:
            logging.info(f'loading supernet {sid}')
            load_state_dict = torch.load(os.path.join('../', args.ckpt_path, f'supernets_{sid}.pt'), map_location='cpu')
        except:
            logging.info(f'could not load supernet {sid}, does not exist')
            break
        load_model = get_new_model().cuda()
        load_model.load_state_dict(load_state_dict['state_dict'])
        load_model.set_arch_parameters(load_state_dict['alpha'])
        load_model.set_encoding(load_state_dict['enc'])
        logging.info(load_model.enc)
        
        #### eval all child models
        results = [] # (opids_str, genotype_str, valid_acc)
        enc = load_model.enc.clone()
        num_edges, num_ops = enc.shape
        for i in range(num_ops**num_edges):
            if (i + 1) % 500 == 0:
                logging.info('---> PROGRESS: {:.2f}%'.format((i + 1)/num_ops**num_edges * 100))
            ## get a child model
            theta_child = torch.zeros_like(enc)
            opids_str = ig_utils.int2base(i, base=num_ops).zfill(num_edges)
            opids = [int(c) for c in opids_str]
            for eid, opid in enumerate(opids):
                theta_child[eid, opid] = 1
            
            if not load_model.is_from_this_supernet(theta_child):
                continue # this arch does not belong to the current supernet

            ## eval
            if load_model.check_connect(theta_child) == False: # broken arch, 10%
                results.append([opids_str, genotype.tostr(), torch.tensor([10]).cuda()])
            genotype, valid_acc = eval_supernet(valid_queue, load_model, criterion, theta_child)
            results.append([opids_str, genotype.tostr(), valid_acc])

        #### write results
        torch.save(results, os.path.join('../', args.ckpt_path, f'rank_{sid}.pt'))

    writer.close()


def eval_supernet(valid_queue, model, criterion, theta):
    ## pre logging
    model._arch_parameters.data.copy_(theta)
    genotype = model.genotype()
    # logging.info('genotype = %s', genotype)

    ## eval
    valid_acc, valid_obj = infer(valid_queue, model, criterion, log=False, eval=False, theta=theta)
    logging.info('valid_acc  %f', valid_acc)
    logging.info('valid_loss %f', valid_obj)

    return genotype, valid_acc


def infer(valid_queue, model, criterion,
          log=True, eval=True, theta=None, double=False, bn_est=False):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()
    model.eval() if eval else model.train() # disable running stats for projection

    if bn_est:
        _data_loader = deepcopy(valid_queue)
        for step, (input, target) in enumerate(_data_loader):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            with torch.no_grad():
                logits = model(input)
        model.eval()

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            if double:
                input = input.double(); target = target.double()
            
            logits = model(input) if theta is None else model(input, theta=theta)
            loss = criterion(logits, target)

            prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)

            if log and step % args.report_freq == 0:
                logging.info('valid %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg))

            if args.fast and step > 1:
                break

    return top1.avg, objs.avg


#### util functions
def distill(result):
    result = result.split('\n')
    cifar10 = result[5].replace(' ', '').split(':')
    cifar100 = result[7].replace(' ', '').split(':')
    imagenet16 = result[9].replace(' ', '').split(':')

    cifar10_train = float(cifar10[1].strip(',test')[-7:-2].strip('='))
    cifar10_test = float(cifar10[2][-7:-2].strip('='))
    cifar100_train = float(cifar100[1].strip(',valid')[-7:-2].strip('='))
    cifar100_valid = float(cifar100[2].strip(',test')[-7:-2].strip('='))
    cifar100_test = float(cifar100[3][-7:-2].strip('='))
    imagenet16_train = float(imagenet16[1].strip(',valid')[-7:-2].strip('='))
    imagenet16_valid = float(imagenet16[2].strip(',test')[-7:-2].strip('='))
    imagenet16_test = float(imagenet16[3][-7:-2].strip('='))

    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


def query(api, genotype, logging):
    result = api.query_by_arch(genotype)
    logging.info('{:}'.format(result))
    cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
        cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = distill(result)
    logging.info('cifar10 train %f test %f' % (cifar10_train, cifar10_test))
    logging.info('cifar100 train %f valid %f test %f' % (cifar100_train, cifar100_valid, cifar100_test))
    logging.info('imagenet16 train %f valid %f test %f' % (imagenet16_train, imagenet16_valid, imagenet16_test))
    return cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
           cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test


if __name__ == '__main__':
    main()

