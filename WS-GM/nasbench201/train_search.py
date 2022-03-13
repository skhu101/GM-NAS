import os
import sys
sys.path.insert(0, '../')
from pprint import pformat
import numpy as np
import torch
import shutil
import nasbench201.utils as ig_utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from nasbench201.search_model_darts import TinyNetworkDarts
from nasbench201.search_model_ws import TinyNetworkWS
from nasbench201.search_model_ws_snas import TinyNetworkSNASWS
from nasbench201.search_model_ws_rsws import TinyNetworkRSWS
from nasbench201.cell_operations import SearchSpaceNames
from nasbench201.architect_ig import Architect
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from nas_201_api import NASBench201API as API
from nasbench201.projection import darts_project, rsws_project


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
parser.add_argument('--supernet_train', type=str, default='darts', choices=['darts', 'rsws', 'snas'])
parser.add_argument('--projection', type=str, default='darts', choices=['darts', 'rsws', 'snas'], help='arch selection method')
#### restart
parser.add_argument('--restart', type=str, default='none', choices=['none', 'all', 'final'],
                    help='reset states after every split (all) or final split (final)')
parser.add_argument('--warmup_epoch', type=int, default=0, help='fix alpha during warmup phase')
#### snas
parser.add_argument('--tau_min', type=float, default=0.1, help='temperature term for snas')
parser.add_argument('--tau_max', type=float, default=1, help='temperature term for snas')
#### dev
parser.add_argument('--hard', type=int, default=1, help='if hard, use one-hot prob during splitting')
parser.add_argument('--fix_sche', type=int, default=0, help='fixing scheduler epoch')
#### optimizer (align with few-shot NAS)
parser.add_argument('--fs_opt', type=int, default=0, help='fs optimizer and batch size (see args augment)')
args = parser.parse_args()


#### args augment
expid = args.save
if 'debug' in args.expid_tag: args.group = 'debug'
if 'dev'   in args.expid_tag: args.group = 'dev'
if args.fs_opt:
    args.nesterov = True
    args.batch_size = 48
    args.weight_decay = 5e-4
if args.supernet_train == 'rsws':
    args.weight_decay = 1e-4

args.save = os.path.join('../experiments/', 'nasbench201', f'{args.group}/', 'search-{}-{}'.format(args.save, args.seed))
if not args.dataset == 'cifar10':
    args.save += '-' + args.dataset
if args.expid_tag != 'none': args.save += '-' + args.expid_tag
args.save += f'_[{args.dis_metric}]'
args.save += f'_[{args.split_crit}]'
args.save += f'_[{args.split_ckpts}|{args.projection_warmup_epoch}]'
args.save += f'_[edge-{args.edge_crit}]'
if args.split_eids != 'none':
    args.save += f'_[{args.split_eids}]'
    args.split_eids = np.array([int(x) for x in args.split_eids.split(',')])
if args.restart != 'none': args.save += f'_[res-{args.restart}]'
args.split_ckpts = [int(x) for x in args.split_ckpts.split(',')]
if args.restart != 'none': args.warmup_epoch = args.split_ckpts[-1]
if args.warmup_epoch > 0: args.save += f'_[wm-{args.warmup_epoch}]'
if args.search_space == 'nas-bench-201': args.save += f'_[fullspace]'
if args.hard == 0: args.save += f'_[soft]'
if args.fs_opt: args.save += f'_[fsopt]'
if args.restart != 'none' and args.fix_sche: args.save += f'_[fix_sche]'

args.epochs = args.split_ckpts[-1] + args.projection_warmup_epoch


#### dir management
## save dir and scripts
if args.resume_epoch > 0: # do not delete dir if resume:
    args.save = '../experiments/nasbench201/{}'.format(args.resume_expid)
    if not os.path.exists(args.save):
        print('no such directory {}'.format(args.save))
else:
    if os.path.exists(args.save):
        if 'debug' in args.expid_tag or input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
            print('proceed to override saving directory')
            shutil.rmtree(args.save)
        else:
            exit(0)
    ig_utils.create_exp_dir(args.save, run_script='../exp_scripts/{}'.format(expid + '.sh'))
## ckpts
args.ckpt_dir = os.path.join(args.save, 'ckpts')
if not os.path.exists(args.ckpt_dir):
    os.mkdir(args.ckpt_dir)
## log file
if args.resume_epoch > 0:
    log_file = 'log_resume-{}_dev-{}_seed-{}_intv-{}'.format(args.resume_epoch, args.dev, args.seed, args.proj_intv)
    if args.log_tag != '': log_file += args.log_tag
else:
    log_file = 'log'
if args.log_tag == 'debug': log_file = 'log_debug'
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
    logging.info("args = %s", args)
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
    elif args.method in ['ws', 'ws-so']:
        helper = lambda: TinyNetworkWS(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args)
    elif args.method == 'ws-snas':
        helper = lambda: TinyNetworkSNASWS(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args)
    elif args.method == 'ws-rsws':
        helper = lambda: TinyNetworkRSWS(C=args.init_channels, N=5, max_nodes=4, num_classes=n_classes, criterion=criterion, search_space=search_space, args=args, track_running_stats=False)
    else:
        logging.info(f'ERROR: method {args.method} not supported'); exit()
    
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

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True)

    #### scheduler
    total_epoch = args.split_ckpts[-1] + args.projection_warmup_epoch
    get_new_scheduler = lambda model, total_epoch: torch.optim.lr_scheduler.CosineAnnealingLR(
                                model.optimizer, float(total_epoch), eta_min=args.learning_rate_min)
    scheduler = get_new_scheduler(model, total_epoch)

    if 'snas' in args.method:
        get_new_temp_scheduler = lambda epochs: ig_utils.TempScheduler(epochs, model.tau, args.tau_max, temp_min=args.tau_min)
        model.temp_scheduler = get_new_temp_scheduler(args.epochs)
    
    
    #### resume (need some adjustment for nas-ws)


    #### supernet training
    ## a supernet is defined by (model, architect, scheduler), the optimizer is part of the model
    if args.split_eids == 'none':
        split_eids = np.random.permutation(model.arch_parameters()[0].shape[0])
    else:
        split_eids = args.split_eids

    supernets = [[model, architect, scheduler]]
    start_epoch = 0
    split_id = 0
    for level, end_epoch in enumerate(args.split_ckpts):
        new_supernets = []
        
        ## train all supernets at current level, and split them along the way
        for cur_model, cur_architect, cur_scheduler in supernets:
            train_supernet(train_queue, valid_queue, cur_model, cur_architect, criterion, cur_scheduler,
                        start_epoch, end_epoch, train_transform, api)

            # split
            if end_epoch == args.split_ckpts[-1] and args.skip_final_split:
                logging.info('Skipping final split for reproducing few-shot NAS')
                continue
            if args.edge_crit == 'predefine':
                split_eid = split_eids[level]
            elif args.edge_crit == 'random-fully': # use randomly generated splits
                split_eid = np.random.choice(cur_model.get_unsplitted_eids().detach().cpu().numpy())
            elif args.edge_crit == 'random-perm': # use different edge every split (regardless of which supernet)
                split_eid = split_eids[split_id]; split_id += 1
            elif args.edge_crit == 'grad':
                split_eid = None
            else:
                logging.info(f'Unrecognized edge criteria: {args.edge_crit}'); exit()

            encs_splitted = split_supernet(train_queue, valid_queue, cur_model, cur_architect, criterion,
                                           split_eid=split_eid, split_crit=args.split_crit, split_num=args.split_num)
            
            # spawn new supernets from the currently trained supernet
            restart = False
            if args.restart == 'all': restart = True
            if args.restart == 'final' and end_epoch == args.split_ckpts[-1]: restart = True

            for enc in encs_splitted:
                sub_model     = get_new_model().cuda()
                sub_model.set_encoding(enc)
                sub_architect = get_new_architect(sub_model)
                if args.fix_sche:
                    sub_scheduler = get_new_scheduler(sub_model, total_epoch - end_epoch)
                else:
                    sub_scheduler = get_new_scheduler(sub_model, total_epoch)
                if 'snas' in args.method:
                    sub_model.temp_scheduler = get_new_temp_scheduler(total_epoch - end_epoch)
                
                if not restart: # load weights only if no restart
                    for p, pp in zip(sub_model.parameters(), cur_model.parameters()):
                        p.data.copy_(pp.data) # safer, or just load_state_dict
                    sub_model.set_arch_parameters(cur_model.arch_parameters())
                    sub_model    .optimizer.load_state_dict(cur_model    .optimizer.state_dict())
                    sub_architect.optimizer.load_state_dict(cur_architect.optimizer.state_dict())
                    sub_scheduler.load_state_dict(cur_scheduler.state_dict())
                    if 'snas' in args.method:
                        sub_model.temp_scheduler.load_state_dict(cur_model.temp_scheduler.state_dict())
                        sub_model.tau = cur_model.tau
                
                new_supernets.append((sub_model, sub_architect, sub_scheduler))
        
        ## move on to the next level
        start_epoch = end_epoch
        if len(new_supernets) > 0:
            supernets = new_supernets


    #### projection warmup (train each supernet for a little while before final projection)
    for cur_model, cur_architect, cur_scheduler in supernets:
        cur_model.mode = args.projection # set the supernet mode to rsws/darts
        train_supernet(train_queue, valid_queue, cur_model, cur_architect, criterion, cur_scheduler,
                       start_epoch, start_epoch + args.projection_warmup_epoch, train_transform, api)

    #### save checkpoint
    for sid, (save_model, _, _) in enumerate(supernets):
        save_state_dict = {
            'state_dict': save_model.state_dict(),
            'enc': save_model.enc,
            'alpha': save_model.arch_parameters(),
        }
        torch.save(save_state_dict, os.path.join(args.ckpt_dir, f'supernets_{sid}.pt'))


    #### architecture selection / projection
    best_valid_acc, best_alpha = 0, None
    for cur_model, cur_architect, _ in supernets:
        logging.info(cur_model.enc)
        if args.projection in ['darts', 'snas']:
            ret = darts_project(train_queue, valid_queue, cur_model, cur_architect, criterion, cur_model.optimizer,
                              start_epoch, args, infer, query)
        elif args.projection == 'rsws':
            ret = rsws_project(train_queue, valid_queue, cur_model, cur_architect, criterion, cur_model.optimizer,
                               start_epoch, args, infer, query)

        valid_acc, alpha, enc = ret
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_alpha = alpha
            best_enc   = enc

    # eval final model
    logging.info('======================== FINAL ========================')
    model = get_new_model().cuda() # get any model
    model.set_encoding(best_enc)
    model.set_arch_parameters([best_alpha]) # arch_parameters is a list
    if not args.fast:
        query(api, model.genotype(), logging)
    writer.close()


def train_supernet(train_queue, valid_queue, model, architect, criterion, scheduler,
                   start_epoch, end_epoch, train_transform, api, **kwargs):
    for epoch in range(start_epoch, end_epoch):
        lr = scheduler.get_lr()[0] # this differs from the actual lr used in optimizer, an issue of pytorch library
        ## data aug
        if args.cutout:
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logging.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
        else:
            logging.info('epoch %d lr %e', epoch, lr)

        ## pre logging
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        model.printing(logging)

        ## training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, model.optimizer, lr, epoch)
        logging.info('train_acc  %f', train_acc)
        logging.info('train_loss %f', train_obj)

        ## eval
        valid_acc, valid_obj = infer(valid_queue, model, criterion, log=False, eval=False)
        logging.info('valid_acc  %f', valid_acc)
        logging.info('valid_loss %f', valid_obj)

        ## logging
        if not args.fast:
            # nasbench201
            cifar10_train, cifar10_test, cifar100_train, cifar100_valid, \
                cifar100_test, imagenet16_train, imagenet16_valid, imagenet16_test = query(api, model.genotype(), logging)

            # tensorboard
            writer.add_scalars('accuracy', {'train':train_acc,'valid':valid_acc}, epoch)
            writer.add_scalars('loss', {'train':train_obj,'valid':valid_obj}, epoch)
            writer.add_scalars('nasbench201/cifar10', {'train':cifar10_train,'test':cifar10_test}, epoch)
            writer.add_scalars('nasbench201/cifar100', {'train':cifar100_train,'valid':cifar100_valid, 'test':cifar100_test}, epoch)
            writer.add_scalars('nasbench201/imagenet16', {'train':imagenet16_train,'valid':imagenet16_valid, 'test':imagenet16_test}, epoch)

        #### scheduling
        scheduler.step()
        if 'snas' in args.method:
            model.tau = model.temp_scheduler.step()
            logging.info('tau %f', model.tau)

        #### saving
        save_state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'alpha': model.arch_parameters(),
            'optimizer': model.optimizer.state_dict(),
            'arch_optimizer': architect.optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }

        if save_state['epoch'] % args.ckpt_interval == 0:
            ig_utils.save_checkpoint(save_state, False, args.ckpt_dir, per_epoch=True)
    return model, architect


# split a supernet into subnets, return encodings of splitted supernet
def split_supernet(train_queue, valid_queue, model, architect, criterion,
                   split_eid, split_crit, split_num):
    num_ops = model.arch_parameters()[0].shape[1]
    
    if split_crit == 'grad':
        eids = torch.where(model.enc.sum(dim=-1) == model.enc.shape[1])[0] if split_eid is None else [split_eid]
        best_edge_score, best_eid, best_groups = 0, 9999, None
        for eid in eids:
            model.train()
            repeat = 100 if not args.fast else 10
            dist_avg = 0
            for _ in range(repeat):
                ## fetch data (one batch for now)
                input, target = next(iter(train_queue))
                input = input.cuda(); target = target.cuda(non_blocking=True)
                input_search, target_search = next(iter(valid_queue))
                input_search = input_search.cuda(); target_search = target_search.cuda(non_blocking=True)
                ## fetch architectures (one arch for now)
                theta = model.get_theta()
                theta_eid_orig = theta[eid].detach().clone()
                theta[eid].data.fill_(0)
                ## get gradients
                split_op_grads = []
                for opid in range(num_ops):
                    theta_op = theta.clone().detach()
                    if args.hard:
                        theta_op[eid, opid] = 1
                    else:
                        theta_op[eid, opid] = 1 * theta_eid_orig[opid]
                    model.optimizer.zero_grad(); architect.optimizer.zero_grad()
                    logits = model(input, theta=theta_op)
                    loss = criterion(logits, target)
                    loss.backward()
                    grads = model.get_split_gradients(split_eid=eid)
                    grads = [g.clone().detach() for g in grads]
                    split_op_grads.append(grads)

                ## compute matching scores (redundant as dist_mat is symmetric, but good for debugging)
                dist_mat = torch.zeros((num_ops, num_ops))
                for opid_i in range(num_ops):
                    for opid_j in range(num_ops):
                        dist_mat[opid_i, opid_j] = ig_utils.match_loss(split_op_grads[opid_i], split_op_grads[opid_j],
                                                                    device=model.arch_parameters()[0].device, dis_metric=args.dis_metric)
                dist_avg += dist_mat
            dist_avg /= repeat
            logging.info('distance matrix:')
            logging.info('\n' + str(dist_avg))

            ## partition
            groups, edge_score = ig_utils.mincut_split_201(dist_avg.numpy(), split_num)

            ## compute edge score
            if edge_score > best_edge_score:
                best_edge_score = edge_score
                best_eid = eid
                best_groups = groups
        split_eid = best_eid
    elif split_crit == 'fewshot': # when num_ops == split_num, reuse random split
        best_groups = ig_utils.random_split_201(split_num, num_ops)
        split_eid = split_eid
    else:
        print(f"ERROR: UNRECOGNIZED SPLIT CRITERIA: {split_crit}"); exit(1)
    
    ## generate encodings
    encs_splitted = []
    for group in best_groups:
        enc = model.enc.clone()
        enc_eid = torch.zeros_like(enc[split_eid])
        enc_eid[torch.LongTensor(group)] = 1
        enc[split_eid] = enc_eid
        encs_splitted.append(enc)
    return encs_splitted


def train(train_queue, valid_queue, model, architect, optimizer, lr, epoch, **kwargs):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()

    for step in range(len(train_queue)):
        model.train()

        ## data
        input, target = next(iter(train_queue))
        input = input.cuda(); target = target.cuda(non_blocking=True)
        input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda(); target_search = target_search.cuda(non_blocking=True)

        ## train alpha
        optimizer.zero_grad(); architect.optimizer.zero_grad()
        shared = None
        if epoch >= args.warmup_epoch:
            shared = architect.step(input, target, input_search, target_search,
                                    eta=lr, model_optimizer=optimizer)

        ## train weight
        optimizer.zero_grad(); architect.optimizer.zero_grad()
        logits, loss = model.step(input, target, args, shared=shared)

        ## logging
        prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f %f' % (step, objs.avg, top1.avg, top5.avg))

        if args.fast:
            break

    return top1.avg, objs.avg


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

