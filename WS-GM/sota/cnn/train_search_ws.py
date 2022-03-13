import os
import sys
sys.path.insert(0, '../../')
import time
import glob
import random
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
from itertools import chain, combinations

from sota.cnn.model_search import Network as DartsNetwork
from sota.cnn.model_search_ws import NetworkWS
from sota.cnn.model_search_ws_snas import NetworkSNASWS
from nasbench201.architect_ig import Architect
from sota.cnn.spaces import spaces_dict

from torch.utils.tensorboard import SummaryWriter


torch.set_printoptions(precision=4, sci_mode=False)

parser = argparse.ArgumentParser("sota")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='choose dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='auto', help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--cutout_prob', type=float, default=1.0, help='cutout probability')
parser.add_argument('--save', type=str, default='exp', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--search_space', type=str, default='s5', help='searching space to choose from')
#### common
parser.add_argument('--ckpt_interval', type=int, default=1, help="interval (epoch) for saving checkpoints")
#parser.add_argument('--method', type=str)
parser.add_argument('--arch_opt', type=str, default='adam', help='architecture optimizer')
parser.add_argument('--resume_epoch', type=int, default=0, help="load ckpt, start training at resume_epoch")
parser.add_argument('--resume_expid', type=str, default='', help="full expid to resume from, name == ckpt folder name")
parser.add_argument('--dev', type=str, default='', help="dev mode")
parser.add_argument('--deter', action='store_true', default=False, help='fully deterministic, for debugging only, slow like hell')
parser.add_argument('--expid_tag', type=str, default='', help="extra tag for expid, 'debug' for debugging")
parser.add_argument('--log_tag', type=str, default='', help="extra tag for log, use 'debug' for debug")
#### darts 2nd order
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
#### sdarts
parser.add_argument('--perturb_alpha', type=str, default='none', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')
#### dev
## common
parser.add_argument('--tune_epochs', type=int, default=140, help='not used for projection (use proj_intv instead)')
parser.add_argument('--fast', action='store_true', default=False, help='eval/train on one batch, for debugging')
parser.add_argument('--dev_resume_epoch', type=int, default=-1, help="resume epoch for arch selection phase, starting from 0")
parser.add_argument('--dev_resume_log', type=str, default='', help="resume log name for arch selection phase")

#### weight-sharing NAS
parser.add_argument('--method', type=str, default='ws', help='choose nas method')
parser.add_argument('--dis_metric', type=str, default='cos', choices=['per-filter-cos', 'cos', 'mse'])
parser.add_argument('--split_eid', type=int, default=0, choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], help='for checking gradient only')
parser.add_argument('--split_ckpts', type=str, default=None, help='e.g. 20,40,60')
parser.add_argument('--skip_final_split', type=int, default=0, help='whether to split at split_ckpts[-1]; used for reproducing few-shot NAS only')
parser.add_argument('--edge_crit', type=str, default='rand', choices=['rand', 'grad'], help='e.g. rand, grad')
parser.add_argument('--split_crit', type=str, default='grad', choices=['grad', 'fewshot'], help='e.g. rand, grad')
parser.add_argument('--split_num', type=int, default=2, choices=[2,7], help='split into how many groups?')
parser.add_argument('--projection_warmup_epoch', type=int, default=0, help='how many epochs to train the supernet before final projection')
parser.add_argument('--supernet_train', type=str, default='darts', choices=['darts', 'pt', 'rsws', 'snas'], help='arch selection method')
parser.add_argument('--projection', type=str, default='darts', choices=['darts', 'pt', 'rsws', 'snas'], help='arch selection method')
#### migrated designs from 201
parser.add_argument('--restart', type=str, default='none', choices=['all', 'final', 'none'], help='reinitialize the model parameters and schedulers')
parser.add_argument('--fix_alpha_equal', action='store_true', default=False, help='')
parser.add_argument('--fix_sche', type=int, default=0, help='fix schedule for restart')

#### ws + gdas
parser.add_argument('--tau_max', type=float, default=1, help='Max temperature (tau) for the gumbel softmax.')
parser.add_argument('--tau_min', type=float, default=0.03, help='Min temperature (tau) for the gumbel softmax.')
args = parser.parse_args()

#### macros


#### args augment
if args.expid_tag != '':
    args.save += '-{}'.format(args.expid_tag)
expid = args.save
args.save = '../../experiments/sota/{}/search-{}-{}-{}'.format(
    args.dataset, args.save, args.search_space, args.seed)

if args.unrolled:
    args.save += '-unrolled'
if not args.weight_decay == 3e-4:
    args.save += '-weight_l2-' + str(args.weight_decay)
if not args.arch_weight_decay == 1e-3:
    args.save += '-alpha_l2-' + str(args.arch_weight_decay)
if args.cutout:
    args.save += '-cutout-' + str(args.cutout_length) + '-' + str(args.cutout_prob)
args.save += f'_[{args.dis_metric}]'
args.save += f'_[e{args.split_eid}]'
args.save += f'_[edge_crit_{args.edge_crit}]'
args.save += f'_[split_crit_{args.split_crit}]'
if args.fix_alpha_equal:
    args.save += '-fix_alpha_equal'
if args.restart != 'none': args.save += f'_[res-{args.restart}]'
if args.fix_sche: args.save += f'_[fix_sche]'
args.split_ckpts = [int(x) for x in args.split_ckpts.split(',')]
args.save += f'_[{args.split_ckpts}|{args.projection_warmup_epoch}]'

args.epochs = args.split_ckpts[-1] + args.projection_warmup_epoch

if args.resume_epoch > 0: # do not delete dir when resume:
    args.save = '../../experiments/sota/{}/{}'.format(args.dataset, args.resume_expid)
    assert(os.path.exists(args.save), 'resume but {} does not exist!'.format(args.save))
else:
    scripts_to_save = glob.glob('*.py') + glob.glob('../../nasbench201/architect*.py') + glob.glob('../../optimizers/darts/architect.py')
    if os.path.exists(args.save):
        if 'debug' in args.expid_tag or input("WARNING: {} exists, override?[y/n]".format(args.save)) == 'y':
            print('proceed to override saving directory')
            shutil.rmtree(args.save)
        else:
            exit(0)
    ig_utils.create_exp_dir(args.save, scripts_to_save=scripts_to_save)

#### logging
log_format = '%(asctime)s %(message)s'
def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p'))
    handler.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger(name)
    logger.addHandler(handler)

    return logger

#### logging
# log_format = '%(asctime)s %(message)s'
# logging.basicConfig(stream=sys.stdout, level=logging.INFO,
#     format=log_format, datefmt='%m/%d %I:%M:%S %p')
log_file = 'log'
if args.resume_epoch > 0:
    log_file += '_resume-{}'.format(args.resume_epoch)
if args.dev_resume_epoch >= 0:
    log_file += '_dev-resume-{}'.format(args.dev_resume_epoch)
if args.dev != '':
    log_file += '_dev-{}'.format(args.dev)
    if args.dev == 'proj':
        log_file += '_intv-{}_ED-{}_PCN-{}_PCR-{}'.format(
                    args.proj_intv, args.edge_decision, args.proj_crit_normal, args.proj_crit_reduce)
    else:
        print('ERROR: DEV METHOD NOT SUPPORTED IN LOGGING:', args.dev); exit(0)
    log_file += '_seed-{}'.format(args.seed)

    if args.log_tag != '': log_file += '_tag-{}'.format(args.log_tag)
if args.log_tag == 'debug': ## prevent redundant debug log files
    log_file = 'log_debug'
log_file += '.txt'
log_path = os.path.join(args.save, log_file)

if args.log_tag != 'debug' and os.path.exists(log_path):
    if input("WARNING: {} exists, override?[y/n]".format(log_file)) == 'y':
        print('proceed to override log file directory')
    else:
        exit(0)

# set up new logger
# logger
logger = setup_logger('logger_main', log_path)
# logger.info('This is just info message')

# warm up arch logger file
log_path_arch = os.path.join(args.save, 'log_arch_update.txt')
logger_arch = setup_logger('logger_arch', log_path_arch)

logger.info('======> log filename: %s', log_file)

# fh = logging.FileHandler(log_path, mode='w')
# fh.setFormatter(logging.Formatter(log_format))
# logging.getLogger().addHandler(fh)


# log_path_arch = os.path.join(args.save, 'warmup_arch_update.txt')
# fh_arch = logging.FileHandler(log_path_arch, mode='w')
# fh_arch.setFormatter(logging.Formatter(log_format))
# logger_arch = logging.getLogger().addHandler(fh_arch)
# logger_arch.setLevel(logging.INFO)
# logger_arch.addHandler(fh_arch)

writer = SummaryWriter(args.save + '/runs')

#### dev resume dir
args.dev_resume_checkpoint_dir = os.path.join(args.save, args.dev_resume_log)
print(args.dev_resume_checkpoint_dir)
if not os.path.exists(args.dev_resume_checkpoint_dir):
    os.mkdir(args.dev_resume_checkpoint_dir)
args.dev_save_checkpoint_dir = os.path.join(args.save, log_file.replace('.txt', ''))
print(args.dev_save_checkpoint_dir)
if not os.path.exists(args.dev_save_checkpoint_dir):
    os.mkdir(args.dev_save_checkpoint_dir)

if args.dataset == 'cifar100':
    n_classes = 100
else:
    n_classes = 10

def main():
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        logger.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    gpu = ig_utils.pick_gpu_lowest_memory() if args.gpu == 'auto' else int(args.gpu)
    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logger.info('gpu device = %d' % gpu)
    logger.info("args = %s", args)

    #### data
    if args.dataset == 'cifar10':
        train_transform, valid_transform = ig_utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        train_transform, valid_transform = ig_utils._data_transforms_cifar100(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'svhn':
        train_transform, valid_transform = ig_utils._data_transforms_svhn(args)
        train_data = dset.SVHN(root=args.data, split='train', download=True, transform=train_transform)
        valid_data = dset.SVHN(root=args.data, split='test', download=True, transform=valid_transform)

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

    test_queue  = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size,
        pin_memory=True)
    
    #### model
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.method in ['ws', 'ws-so']:
        helper = lambda: NetworkWS(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], args)
    elif args.method == 'ws-snas':
        helper = lambda: NetworkSNASWS(args.init_channels, n_classes, args.layers, criterion, spaces_dict[args.search_space], args)

    def get_new_model():
        model = helper()
        model.get_new_model = helper
        return model

    model = get_new_model()
    model = model.cuda()
    logger.info("param size = %fMB", ig_utils.count_parameters_in_MB(model))
    
    get_new_architect = lambda model: Architect(model, args)
    architect = Architect(model, args)

    #### scheduler
    get_new_scheduler = lambda model, epoch: torch.optim.lr_scheduler.CosineAnnealingLR(
                                model.optimizer, float(epoch), eta_min=args.learning_rate_min)
    scheduler = get_new_scheduler(model, args.epochs)

    if 'snas' in args.method:
        get_new_temp_scheduler = lambda epochs: ig_utils.TempScheduler(epochs, model.tau, args.tau_max, temp_min=args.tau_min)
        model.temp_scheduler = get_new_temp_scheduler(args.epochs)

    ## a supernet is defined by (model, architect, scheduler), the optimizer is part of the model
    split_eids = np.random.permutation(model.arch_parameters()[0].shape[0])
    supernets = [[model, architect, scheduler]]
    start_epoch = 0 
    split_id = 0 
    for end_epoch in args.split_ckpts:
        new_supernets = []
        count = 0
    
        ## train all supernets at current level, and split them along the way
        for cur_model, cur_architect, cur_scheduler in supernets:
            train_supernet(train_queue, valid_queue, cur_model, cur_architect, criterion, cur_scheduler,
                           start_epoch, end_epoch, train_transform)

            # split
            #if end_epoch == args.split_ckpts[-1] and args.skip_final_split:
            #    logger.info('Skipping final split for reproducing few-shot NAS')
            #    continue            
            if args.split_crit == 'grad' and args.edge_crit == 'rand':
                split_eid = split_eids[split_id]
            elif args.split_crit == 'grad' and args.edge_crit == 'grad':
                split_eid = None
            elif args.split_crit == 'fewshot':
                split_eid = args.split_eid
            split_id += 1
            encs_normal_splitted, encs_reduce_splitted = split_supernet(train_queue, valid_queue, cur_model, cur_architect, criterion, get_new_model, split_eid=split_eid, split_crit=args.split_crit, split_num=args.split_num)
    
            # spawn new supernets from the currently trained supernet
            for enc_normal, enc_reduce in zip(encs_normal_splitted, encs_reduce_splitted):
                sub_model     = get_new_model().cuda()
                #sub_model     = get_new_model()

                restart = False
                if args.restart == 'all': restart = True
                if args.restart == 'final' and end_epoch == args.split_ckpts[-1]: restart = True

                if not restart:
                    for p, pp in zip(sub_model.parameters(), cur_model.parameters()):
                        p.data.copy_(pp.data) # safer, or just load_state_dict, need double check
                    sub_model.set_arch_parameters(cur_model.arch_parameters())
                    sub_model.optimizer.load_state_dict(cur_model    .optimizer.state_dict())

                # copy sub_architect and sub_scheduler
                sub_architect = get_new_architect(sub_model)
                if args.fix_sche:
                    sub_scheduler = get_new_scheduler(sub_model, args.epochs - end_epoch)
                    print(args.epochs - end_epoch)
                else:
                    sub_scheduler = get_new_scheduler(sub_model, args.epochs)
                if 'snas' in args.method:
                    sub_model.temp_scheduler = get_new_temp_scheduler(args.epochs - end_epoch)

                if not restart:
                    sub_architect.optimizer.load_state_dict(cur_architect.optimizer.state_dict())
                    sub_scheduler.load_state_dict(cur_scheduler.state_dict())
                    if 'snas' in args.method:
                        sub_model.temp_scheduler.load_state_dict(cur_model.temp_scheduler.state_dict())
                        sub_model.tau = cur_model.tau

                new_supernets.append((sub_model, sub_architect, sub_scheduler))
    
                sub_model.set_encoding(enc_normal, enc_reduce)
                logger.info('\n' + 'sub_model id: ' + str(count))
                logger.info('\n' + 'enc_normal:')
                logger.info('\n' + str(sub_model.enc_normal))
                logger.info('\n' + 'enc_reduce:')
                logger.info('\n' + str(sub_model.enc_reduce))
                count += 1

        ## move on to the next level
        start_epoch = end_epoch
        supernets = new_supernets

    #### projection warmup (train each supernet for a little while before final projection)
    args.fix_alpha_equal = False
    for start_epoch in range(start_epoch, start_epoch+args.projection_warmup_epoch):
        count = 0
        for cur_model, cur_architect, cur_scheduler in supernets:
            cur_model.mode = args.projection # set the supernet mode to rsws/darts
            train_supernet(train_queue, valid_queue, cur_model, cur_architect, criterion, cur_scheduler,
                           start_epoch, start_epoch + 1, train_transform, cur_model_id=count, logger_arch=logger_arch)
            count += 1

    #### architecture partition
    count = 0
    for cur_model, cur_architect, cur_scheduler in supernets:
        logger.info('\n' + 'sub_model id: ' + str(count))
        ## validation
        valid_acc, valid_obj = infer(valid_queue, cur_model, log=False)
        logger.info('valid_acc %f | valid_obj %f', valid_acc, valid_obj)

        test_acc, test_obj = infer(test_queue, cur_model, log=False)
        logger.info('test_acc %f | test_obj %f', test_acc, test_obj)

        if args.projection == 'darts':
            logger.info('genotype: ')
            logger.info('\n' + str(cur_model.genotype()))
        
        save_state_dict = {
            'epoch': start_epoch + args.projection_warmup_epoch + 1,
            'state_dict': cur_model.state_dict(),
            'alpha': cur_model.arch_parameters(),
            'optimizer': cur_model.optimizer.state_dict(),
            'arch_optimizer': cur_architect.optimizer.state_dict(),
            'scheduler': cur_scheduler.state_dict()
        }
        prefix_name = 'submodel_'+str(count)+'_'
        ig_utils.save_checkpoint(save_state_dict, False, args.save, per_epoch=False, prefix=prefix_name)
        
        count += 1

    writer.close()

def train_supernet(train_queue, valid_queue, model, architect, criterion, scheduler,
                   start_epoch, end_epoch, train_transform, cur_model_id=-1, logger_arch=None):
    for epoch in range(start_epoch, end_epoch):
        lr = scheduler.get_lr()[0]
        ## data aug
        if args.cutout:
            train_transform.transforms[-1].cutout_prob = args.cutout_prob * epoch / (args.epochs - 1)
            logger.info('epoch %d lr %e cutout_prob %e', epoch, lr,
                         train_transform.transforms[-1].cutout_prob)
            logger.info('epoch %d arch_lr %e', epoch, architect.optimizer.param_groups[0]['lr'])
        else:
            logger.info('epoch %d lr %e', epoch, lr)
            logger.info('epoch %d arch_lr %e', epoch, architect.optimizer.param_groups[0]['lr'])

        if 'snas' in args.method:
            logger.info('epoch %d tau %f', epoch, model.tau)

        ## pre logging
        num_params = ig_utils.count_parameters_in_Compact(model)
        genotype = model.genotype()
        logger.info('param size = %f', num_params)
        logger.info('genotype = %s', genotype)
        model.printing(logging)

        ## training
        train_acc, train_obj = train(train_queue, valid_queue, model, architect, model.optimizer, lr, epoch)
        logger.info('train_acc  %f', train_acc)
        logger.info('train_loss %f', train_obj)

        ## eval
        valid_acc, valid_obj = infer(valid_queue, model, log=False)
        logger.info('valid_acc  %f', valid_acc)
        logger.info('valid_loss %f', valid_obj)

        ## logging
        if not args.fast:
            # tensorboard
            writer.add_scalars('accuracy', {'train':train_acc,'valid':valid_acc}, epoch)
            writer.add_scalars('loss', {'train':train_obj,'valid':valid_obj}, epoch)

        ## scheduler updates (before saving ckpts)
        scheduler.step()
        if 'snas' in args.method:
            model.tau = model.temp_scheduler.step()

        ## saving
        if cur_model_id >= 0:
            if (epoch + 1) % args.ckpt_interval == 0:
                save_state_dict = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'alpha': model.arch_parameters(),
                    'optimizer': model.optimizer.state_dict(),
                    'arch_optimizer': architect.optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
                #ig_utils.save_checkpoint(save_state_dict, False, args.save, per_epoch=True)
                prefix_name = 'submodel_'+str(cur_model_id)+'_'
                ig_utils.save_checkpoint(save_state_dict, False, args.save, per_epoch=True, prefix=prefix_name)

            if logger_arch is not None:
                logger_arch.info('\n' + 'sub_model id ' + str(cur_model_id) + ' epoch ' + str(epoch))
                ## validation
                valid_acc, valid_obj = infer(valid_queue, model, log=False)
                logger_arch.info('valid_acc %f | valid_obj %f', valid_acc, valid_obj)

                #test_acc, test_obj = infer(test_queue, model, log=False)
                #logger_arch.info('test_acc %f | test_obj %f', test_acc, test_obj)

                if args.projection == 'darts':
                    logger_arch.info('genotype: ')
                    logger_arch.info('\n' + str(model.genotype()))

    return model, architect


# split a supernet into subnets, return encodings of splitted supernet
def split_supernet(train_queue, valid_queue, model, architect, criterion, get_new_model, split_eid, split_crit, split_num):
    model.train()
    num_ops = model.arch_parameters()[0].shape[1]
    if split_crit == 'grad':
        eids = torch.where(model.enc_normal.sum(dim=-1) == model.enc_normal.shape[1])[0] if split_eid is None else [split_eid]
        print(eids)
        best_edge_score, best_eid, best_groups = 10000, 9999, None
        for eid in eids:
            repeat = 100 if not args.fast else 1
            dist_avg = 0
            split_normal = True
            split_eid = eid
            for _ in range(repeat):
                encs = [None, None]
                ## fetch data (one batch for now)
                input, target = next(iter(train_queue))
                input = input.cuda(); target = target.cuda(non_blocking=True)
                input_search, target_search = next(iter(valid_queue))
                input_search = input_search.cuda(); target_search = target_search.cuda(non_blocking=True)
                ## fetch architectures (one arch for now)
                weights_normal, weights_reduce = model.get_theta()
                weights_normal_cp = weights_normal.clone(); weights_reduce_cp = weights_reduce.clone() 
                if split_eid < model.arch_parameters()[0].shape[0]:
                    weights_normal[split_eid].data.fill_(0)
                    split_normal = True
                else:
                    weights_reduce[split_eid-model.arch_parameters()[0].shape[0]].data.fill_(0)
                    split_normal = False
                ## get gradients
                split_op_grads = []
                for opid in range(num_ops):
                    if split_normal:
                        if model.mode == 'darts' or 'snas' in model.mode:
                            weights_normal_op = weights_normal.clone(); weights_normal_op[split_eid, opid] = weights_normal_cp[split_eid, opid]
                        elif model.mode == 'rsws':
                            weights_normal_op = weights_normal.clone(); weights_normal_op[split_eid, opid] = 1
                        model.optimizer.zero_grad(); architect.optimizer.zero_grad()
                        logits = model(input, weights_normal=weights_normal_op.data, weights_reduce=weights_reduce.data)
                    else:
                        if model.supernet_train == 'darts' or 'snas' in model.mode:
                            weights_reduce_op = weights_reduce.clone(); weights_reduce_op[split_eid-model.arch_parameters()[0].shape[0], opid] = weights_reduce_cp[split_eid-model.arch_parameters()[0].shape[0], opid].data
                        elif model.supernet_train == 'rsws':
                            weights_reduce_op = weights_reduce.clone(); weights_reduce_op[split_eid-model.arch_parameters()[0].shape[0], opid] = 1
                        model.optimizer.zero_grad(); architect.optimizer.zero_grad()
                        logits = model(input, weights_normal=weights_normal.data, weights_reduce=weights_reduce_op.data)
                    #theta_op = theta.clone(); theta_op[split_eid, opid] = 1
                    #model.optimizer.zero_grad(); architect.optimizer.zero_grad()
                    #logits = model(input, theta=theta_op)
                    loss = criterion(logits, target)
                    loss.backward()
                    grads = model.get_split_gradients(split_eid=split_eid, split_normal=split_normal)
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
            logger.info('\n' + 'split_normal: ' + str(split_normal))
            logger.info('\n' + 'distance matrix:')
            logger.info('\n' + str(dist_avg))

            ## partition
            groups, edge_score = mincut_split_darts(dist_avg.numpy(), split_num) # TODO implement the max-cut algorithm to split the supernet
            print(groups, edge_score)

            ## compute edge score
            if edge_score < best_edge_score:
                best_edge_score = edge_score
                best_eid = eid
                best_groups = groups
        split_eid = best_eid
        groups = best_groups

    elif split_crit == 'fewshot': # when num_ops == split_num, reuse random split
        split_normal = True
        groups = random_split_darts(split_num, num_ops)
    else:
        print(f"ERROR: UNRECOGNIZED SPLIT CRITERIA: {split_crit}"); exit(1)


    encs_normal_splitted = []
    encs_reduce_splitted = []
    if split_normal:
        for group in groups:
            # enc_normal
            enc_normal = model.enc_normal.clone()
            enc_normal_eid = torch.zeros_like(enc_normal[split_eid])
            enc_normal_eid[torch.LongTensor(group)] = 1
            enc_normal[split_eid] = enc_normal_eid
            encs_normal_splitted.append(enc_normal)
            # enc_reduce
            enc_reduce = model.enc_reduce.clone()
            encs_reduce_splitted.append(enc_reduce)        
    else:
        for group in groups:
            # enc_reduce
            enc_reduce = model.enc_reduce.clone()
            enc_reduce_eid = torch.zeros_like(enc_reduce[split_eid-model.arch_parameters()[0].shape[0]])
            enc_reduce_eid[torch.LongTensor(group)] = 1
            enc_reduce[split_eid-model.arch_parameters()[0].shape[0]] = enc_reduce_eid
            encs_reduce_splitted.append(enc_reduce)
            # enc_normal
            enc_normal = model.enc_normal.clone()
            encs_normal_splitted.append(enc_normal)        
    return encs_normal_splitted, encs_reduce_splitted


def train(train_queue, valid_queue, model, architect, optimizer, lr, epoch):
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

        optimizer.zero_grad()
        ## train alpha
        if not args.fix_alpha_equal:
            architect.optimizer.zero_grad()
            architect.step(input, target, input_search, target_search,
                           eta=lr, model_optimizer=optimizer)

        ## sdarts
        #if perturb_alpha:
        #    # transform arch_parameters to prob (for perturbation)
        #    model.softmax_arch_parameters()
        #    optimizer.zero_grad(); architect.optimizer.zero_grad()
        #    perturb_alpha(model, input, target, epsilon_alpha)

        ## train weights
        optimizer.zero_grad(); architect.optimizer.zero_grad()
        logits, loss = model.step(input, target, args)
        
        ## sdarts
        #if perturb_alpha:
        #    ## restore alpha to unperturbed arch_parameters
        #    model.restore_arch_parameters()

        ## logging
        n = input.size(0)
        prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        if step % args.report_freq == 0:
            logger.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

        if args.fast and step >= 50:
            break

    return  top1.avg, objs.avg


def infer(valid_queue, model, log=True, _eval=True, weights_dict=None):
    objs = ig_utils.AvgrageMeter()
    top1 = ig_utils.AvgrageMeter()
    top5 = ig_utils.AvgrageMeter()
    model.eval() if _eval else model.train() # disable running stats for projection

    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            
            if weights_dict is None:
                loss, logits = model._loss(input, target, return_logits=True)
            else:
                logits = model(input, weights_dict=weights_dict)
                loss = model._criterion(logits, target)

            prec1, prec5 = ig_utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if step % args.report_freq == 0 and log:
                logger.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

            if args.fast:
                break

    return top1.avg, objs.avg

def mincut_split_darts(dist_avg, split_num): # note: this is not strictly mincut, but it's fine for 201
    assert split_num == 2, 'always split into 2 groups for darts space (when using gradient to split)'
    assert isinstance(dist_avg, np.ndarray)
  
    vertex = [i for i in range(dist_avg.shape[0])]

    max_cut = 100000
    for subset in chain(*map(lambda x: combinations(vertex, x), range(1, len(vertex)+1))):
        if len(subset) >= 2 and len(subset)<= len(vertex)//2:
            cut = 0 
            for edge in combinations(vertex, 2): 
                if (edge[0] in subset and edge[1] in subset):
                    cut += dist_avg[edge[0], edge[1]]
                if (edge[0] not in subset and edge[1] not in subset):
                    cut += dist_avg[edge[0], edge[1]]
            if cut < max_cut:
                group0 = np.array([ i for i in vertex if i in subset])
                group1 = np.array([ i for i in vertex if i not in subset])
                max_cut = cut 
    best_groups = [group0, group1]
    return best_groups, max_cut

def random_split_darts(split_num, num_ops): # when split_num == num_ops -> split every operation like few-shot NAS
    assert num_ops % split_num == 0, 'always split into even groups'
    if split_num == num_ops: # exhaustive split
        opids = np.arange(0, num_ops)
    else:
        opids = np.random.permutation(num_ops)
    group_size = num_ops // split_num
    groups = [opids[s:s+group_size] for s in np.arange(0, num_ops, group_size)]

    return groups

if __name__ == '__main__':
    main()
