# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import argparse

from models import ImagenetRunConfig
from nas_manager import *
# from models.super_nets.few_shot_super_proxyless import SuperProxylessNASNets
from models.super_nets.super_proxyless import SuperProxylessNASNets

try:
    import moxing as mox
except Exception as e:
    pass

# ref values
ref_values = {
    'flops': {
        '0.35': 59 * 1e6,
        '0.50': 97 * 1e6,
        '0.75': 209 * 1e6,
        '1.00': 300 * 1e6,
        '1.30': 509 * 1e6,
        '1.40': 582 * 1e6,
    },
    # ms
    'mobile': {
        '1.00': 80,
    },
    'cpu': {},
    'gpu8': {},
}

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default=None)
parser.add_argument('--gpu', help='gpu available', default='0,1,2,3,4,5,6,7')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--debug', help='freeze the weight parameters', action='store_true')
parser.add_argument('--manual_seed', default=0, type=int)

""" run config """
parser.add_argument('--n_epochs', type=int, default=120)
parser.add_argument('--init_lr', type=float, default=0.025)
parser.add_argument('--lr_schedule_type', type=str, default='cosine')
# lr_schedule_param

parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet'])
parser.add_argument('--train_batch_size', type=int, default=256)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--valid_size', type=int, default=10000)

parser.add_argument('--opt_type', type=str, default='sgd', choices=['sgd'])
parser.add_argument('--momentum', type=float, default=0.9)  # opt_param
parser.add_argument('--no_nesterov', action='store_true')  # opt_param
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--label_smoothing', type=float, default=0.1)
parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
parser.add_argument('--init_div_groups', action='store_true')
parser.add_argument('--validation_frequency', type=int, default=1)
parser.add_argument('--print_frequency', type=int, default=100)

parser.add_argument('--n_worker', type=int, default=32)
parser.add_argument('--resize_scale', type=float, default=0.08)
parser.add_argument('--distort_color', type=str, default='normal', choices=['normal', 'strong', 'None'])

""" net config """
parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320')
parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')
parser.add_argument('--width_mult', type=float, default=1.0)
parser.add_argument('--bn_momentum', type=float, default=0.1)
parser.add_argument('--bn_eps', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0)

# architecture search config
""" arch search algo and warmup """
parser.add_argument('--arch_algo', type=str, default='grad', choices=['grad', 'rl'])
parser.add_argument('--warmup_epochs', type=int, default=40)
""" shared hyper-parameters """
parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
parser.add_argument('--arch_opt_type', type=str, default='adam', choices=['adam'])
parser.add_argument('--arch_lr', type=float, default=1e-3)
parser.add_argument('--arch_adam_beta1', type=float, default=0)  # arch_opt_param
parser.add_argument('--arch_adam_beta2', type=float, default=0.999)  # arch_opt_param
parser.add_argument('--arch_adam_eps', type=float, default=1e-8)  # arch_opt_param
parser.add_argument('--arch_weight_decay', type=float, default=0)
parser.add_argument('--target_hardware', type=str, default=None, choices=['mobile', 'cpu', 'gpu8', 'flops', None])
""" Grad hyper-parameters """
parser.add_argument('--grad_update_arch_param_every', type=int, default=5)
parser.add_argument('--grad_update_steps', type=int, default=1)
parser.add_argument('--grad_binary_mode', type=str, default='full_v2', choices=['full_v2', 'full', 'two'])
parser.add_argument('--grad_data_batch', type=int, default=None)
parser.add_argument('--grad_reg_loss_type', type=str, default='mul#log', choices=['add#linear', 'mul#log'])
parser.add_argument('--grad_reg_loss_lambda', type=float, default=1e-1)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_alpha', type=float, default=0.2)  # grad_reg_loss_params
parser.add_argument('--grad_reg_loss_beta', type=float, default=0.3)  # grad_reg_loss_params
""" RL hyper-parameters """
parser.add_argument('--rl_batch_size', type=int, default=10)
parser.add_argument('--rl_update_per_epoch', action='store_true')
parser.add_argument('--rl_update_steps_per_epoch', type=int, default=300)
parser.add_argument('--rl_baseline_decay_weight', type=float, default=0.99)
parser.add_argument('--rl_tradeoff_ratio', type=float, default=0.1)
parser.add_argument('--fixed_op', type=str, default='3x3_MBConv3')

#### weight-sharing NAS
parser.add_argument('--dis_metric', type=str, default='cos', choices=['per-filter-cos', 'cos', 'mse'])
parser.add_argument('--split_eid', type=int, default=0, choices=[0,1,2,3,4,5,6,7,8,9,10,11,12,13], help='for checking gradient only')
parser.add_argument('--split_ckpts', type=str, default=None, help='e.g. 20,40,60')
parser.add_argument('--skip_final_split', type=int, default=0, help='whether to split at split_ckpts[-1]; used for reproducing few-shot NAS only')
parser.add_argument('--split_crit', type=str, default='grad', choices=['grad', 'fewshot'], help='e.g. 20,40,60')
parser.add_argument('--edge_crit', type=str, default='rand', choices=['grad', 'rand'], help='e.g. 20,40,60')
parser.add_argument('--split_num', type=str, default=None, help='split into how many groups?')
parser.add_argument('--select_edge_1', help='select edge 1 for splitting', action='store_true')


if __name__ == '__main__':
    args = parser.parse_args()

    args.split_ckpts = [int(x) for x in args.split_ckpts.split(',')]
    args.split_num = [int(x) for x in args.split_num.split(',')]

    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    np.random.seed(args.manual_seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.makedirs(args.path, exist_ok=True)

    # build run config from args
    args.lr_schedule_param = None
    args.opt_param = {
        'momentum': args.momentum,
        'nesterov': not args.no_nesterov,
    }
    run_config = ImagenetRunConfig(
        **args.__dict__
    )

    # debug, adjust run_config
    if args.debug:
        run_config.train_batch_size = 256
        run_config.test_batch_size = 256
        run_config.valid_size = 256
        run_config.n_worker = 0

    width_stages_str = '-'.join(args.width_stages.split(','))
    # build net from args
    args.width_stages = [int(val) for val in args.width_stages.split(',')]
    args.n_cell_stages = [int(val) for val in args.n_cell_stages.split(',')]
    args.stride_stages = [int(val) for val in args.stride_stages.split(',')]
    args.conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
    ]
    # super_net = SuperProxylessNASNets(
    #     width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
    #     conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
    #     bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout, fixed_op=args.fixed_op
    # )
    get_new_model  = lambda: SuperProxylessNASNets(
        width_stages=args.width_stages, n_cell_stages=args.n_cell_stages, stride_stages=args.stride_stages,
        conv_candidates=args.conv_candidates, n_classes=run_config.data_provider.n_classes, width_mult=args.width_mult,
        bn_param=(args.bn_momentum, args.bn_eps), dropout_rate=args.dropout
    )
    super_net = get_new_model()

    # build arch search config from args
    if args.arch_opt_type == 'adam':
        args.arch_opt_param = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps,
        }
    else:
        args.arch_opt_param = None
    if args.target_hardware is None:
        args.ref_value = None
    else:
        args.ref_value = ref_values[args.target_hardware]['%.2f' % args.width_mult]
    if args.arch_algo == 'grad':
        from nas_manager import GradientArchSearchConfig
        if args.grad_reg_loss_type == 'add#linear':
            args.grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
        elif args.grad_reg_loss_type == 'mul#log':
            args.grad_reg_loss_params = {
                'alpha': args.grad_reg_loss_alpha,
                'beta': args.grad_reg_loss_beta,
            }
        else:
            args.grad_reg_loss_params = None
        arch_search_config = GradientArchSearchConfig(**args.__dict__)
    elif args.arch_algo == 'rl':
        from nas_manager import RLArchSearchConfig
        arch_search_config = RLArchSearchConfig(**args.__dict__)
    else:
        raise NotImplementedError

    print('Run config:')
    for k, v in run_config.config.items():
        print('\t%s: %s' % (k, v))
    print('Architecture Search config:')
    for k, v in arch_search_config.config.items():
        print('\t%s: %s' % (k, v))

    # arch search run manager
    # arch_search_run_manager = ArchSearchRunManager(args.path, super_net, run_config, arch_search_config)

    get_new_ArchSearchRunManager = lambda super_net: ArchSearchRunManager(args.path, super_net, run_config, arch_search_config)
    arch_search_run_manager = get_new_ArchSearchRunManager(super_net)

    # resume
    if args.resume:
        try:
            arch_search_run_manager.load_model()
        except Exception:
            from pathlib import Path
            home = str(Path.home())
            warmup_path = str(args.path) + '/checkpoint/warmup.pth.tar'
            # warmup_path = os.path.join(
            #     home, 'Workspace/Exp/arch_search/%s_ProxylessNAS_%.2f_%s/warmup.pth.tar' %
            #           (run_config.dataset, args.width_mult, width_stages_str)
            # )
            if os.path.exists(warmup_path):
                print('load warmup weights')
                arch_search_run_manager.load_model(model_fname=warmup_path)
            else:
                print('fail to load models')

    # warmup_path = 'warmup.pth.tar'
    # arch_search_run_manager.load_model(model_fname=warmup_path)

    # warmup
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)

    warmup_path = 'warmup.pth.tar'
    arch_search_run_manager.load_model(model_fname=warmup_path)

    if args.select_edge_1:
        split_eids = [1]
        split_eids.extend(np.random.permutation(range(2, 22)))
    else:
        split_eids = []
        split_eids.extend(np.random.permutation(range(1, 22)))
    supernets_run_manager = [[super_net, arch_search_run_manager]]
    start_epoch = 0
    split_id = 0

    for index, end_epoch in enumerate(args.split_ckpts):
        new_supernets_run_manager = []
        count = 0
        split_num = args.split_num[index]

        ## train all supernets at current level, and split them along the way
        for cur_model, cur_arch_search_run_manager in supernets_run_manager:
            cur_arch_search_run_manager.run_manager.start_epoch = start_epoch
            cur_arch_search_run_manager.run_manager.end_epoch = end_epoch

            # joint training
            cur_arch_search_run_manager.train(fix_net_weights=args.debug)

            if args.split_crit == 'grad':
                if args.edge_crit == 'rand':
                    split_eid = split_eids[split_id]
                elif args.edge_crit == 'grad':
                    split_eid = None
            elif args.split_crit == 'fewshot':
                split_eid = args.split_eid

            if args.select_edge_1 and split_id ==0 :
                split_eid = split_eids[split_id]


            encs_splitted, select_eid = cur_arch_search_run_manager.split_supernet(get_new_model, get_new_ArchSearchRunManager,
                                           split_eid=split_eid, split_crit=args.split_crit, split_num=split_num, dis_metric=args.dis_metric)

            # spawn new supernets from the currently trained supernet
            for enc in encs_splitted:
                # copy sub_architect and sub_scheduler
                sub_model     = get_new_model()


                cur_arch_search_run_manager.run_manager.save_model({
                    'warmup': False,
                    'weight_optimizer': cur_arch_search_run_manager.run_manager.optimizer.state_dict(),
                    'arch_optimizer': cur_arch_search_run_manager.arch_optimizer.state_dict(),
                    'state_dict': cur_arch_search_run_manager.net.state_dict()
                }, model_name='ckpt_'+str(end_epoch)+'_id_'+str(count)+'_checkpoint.pth.tar')


                # copy encoding
                for id in range(1,22):
                    sub_model.blocks[id].mobile_inverted_conv.enc.copy_(cur_model.blocks[id].mobile_inverted_conv.enc)

                sub_arch_search_run_manager = get_new_ArchSearchRunManager(sub_model)

                sub_arch_search_run_manager.load_model(model_fname=str(args.path) + '/checkpoint/ckpt_'+str(end_epoch)+'_id_'+str(count)+'_checkpoint.pth.tar')

                new_supernets_run_manager.append((sub_model, sub_arch_search_run_manager))

                sub_model.set_encoding(select_eid, enc)

                print('\n' + 'sub_model id: ' + str(count))
                for id in range(1,22):
                    print('\n' + str(sub_model.blocks[id].mobile_inverted_conv.enc))
                count += 1

            split_id += 1

            del cur_model, cur_arch_search_run_manager

        ## move on to the next level
        start_epoch = end_epoch
        supernets_run_manager = new_supernets_run_manager


    ## train all supernets at current level, and split them along the way
    count = 0
    for cur_model, cur_arch_search_run_manager in supernets_run_manager:
        print('\n' + 'sub_model id: ' + str(count))
        cur_arch_search_run_manager.run_manager.start_epoch = start_epoch
        cur_arch_search_run_manager.run_manager.end_epoch = args.n_epochs

        # joint training
        cur_arch_search_run_manager.train(fix_net_weights=args.debug, cur_model_id=count)
        count += 1




