# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import json
import torch.nn as nn
from tqdm import tqdm
import random
import os
import time
import numpy as np
from itertools import chain, combinations, permutations

import torch
import torch.nn.functional as F
# import horovod.torch as hvd
import distributed
from imagenet_codebase.run_manager import DistributedMetric
from imagenet_codebase.utils import accuracy, list_mean, cross_entropy_loss_with_soft_target, subset_mean, \
    AverageMeter, int2list, download_url, mix_images, mix_labels
from imagenet_codebase.data_providers.base_provider import MyRandomResizedCrop
from imagenet_codebase.run_manager.distributed_run_manager import DistributedRunManager


def validate(run_manager, epoch=0, is_test=True, image_size_list=None,
             width_mult_list=None, ks_list=None, expand_ratio_list=None, depth_list=None, additional_setting=None):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    if image_size_list is None:
        image_size_list = int2list(run_manager.run_config.data_provider.image_size, 1)
    if width_mult_list is None:
        width_mult_list = [i for i in range(len(dynamic_net.width_mult_list))]
    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list

    subnet_settings = []
    for w in width_mult_list:
        for d in depth_list:
            for e in expand_ratio_list:
                for k in ks_list:
                    for img_size in image_size_list:
                        subnet_settings.append([{
                            'image_size': img_size,
                            'wid': w,
                            'd': d,
                            'e': e,
                            'ks': k,
                        }, 'R%s-W%s-D%s-E%s-K%s' % (img_size, w, d, e, k)])
    if additional_setting is not None:
        subnet_settings += additional_setting

    losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []

    valid_log = ''
    for setting, name in subnet_settings:
        print(setting)
        run_manager.write_log('-' * 30 + ' Validate %s ' % name + '-' * 30, 'train', should_print=True)
        run_manager.run_config.data_provider.assign_active_img_size(setting.pop('image_size'))
        # setting.pop('image_size')
        dynamic_net.set_active_subnet(**setting)
        run_manager.write_log(dynamic_net.module_str, 'train', should_print=True)

        run_manager.reset_running_statistics(dynamic_net)
        loss, top1, top5 = run_manager.validate(epoch=epoch, is_test=is_test, run_str=name, net=dynamic_net)
        losses_of_subnets.append(loss)
        top1_of_subnets.append(top1)
        top5_of_subnets.append(top5)
        valid_log += '%s (%.3f), ' % (name, top1)

    return list_mean(losses_of_subnets), list_mean(top1_of_subnets), list_mean(top5_of_subnets), valid_log


# different submodels are sampled during each step, many submodels are trained in one epoch
# total num of epochs = warmup_epochs + num_epoch
# dynamic_batchsize sets the number of different models trained in each step(using the same data), loss is then backpropped together
def train_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.net

    # switch to train mode
    dynamic_net.train()
    run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric('train_loss')
    top1 = DistributedMetric('train_top1')
    top5 = DistributedMetric('train_top5')

    # with tqdm(total=nBatch,
    #           desc='Train Epoch #{}'.format(epoch + 1),
    #           disable=not run_manager.is_root) as t:
    end = time.time()
    for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
        data_time.update(time.time() - end)
        if epoch < warmup_epochs:
            new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                run_manager.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
            )
        else:
            new_lr = run_manager.run_config.adjust_learning_rate(
                run_manager.optimizer, epoch - warmup_epochs, i, nBatch
            )

        images, labels = images.cuda(), labels.cuda()
        target = labels

        # soft target
        if args.kd_ratio > 0:
            args.teacher_model.train()
            with torch.no_grad():
                soft_logits = args.teacher_model(images).detach()
                soft_label = F.softmax(soft_logits, dim=1)

        # clear gradients
        run_manager.optimizer.zero_grad()

        loss_of_subnets, acc1_of_subnets, acc5_of_subnets = [], [], []
        # compute output
        subnet_str = ''
        for _ in range(args.dynamic_batch_size):

            # set random seed before sampling
            if args.independent_distributed_sampling:
                subnet_seed = os.getpid() + time.time()
            else:
                subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, _, 0))
            random.seed(subnet_seed)
            subnet_settings = dynamic_net.sample_active_subnet()
            subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
            ) for key, val in subnet_settings.items()]) + ' || '

            output = run_manager.net(images)
            if args.kd_ratio == 0:
                loss = run_manager.train_criterion(output, labels)
                loss_type = 'ce'
            else:
                if args.kd_type == 'ce':
                    kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                else:
                    kd_loss = F.mse_loss(output, soft_logits)
                loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
                loss = loss * (2 / (args.kd_ratio + 1))
                loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss_of_subnets.append(loss)
            acc1_of_subnets.append(acc1[0])
            acc5_of_subnets.append(acc5[0])
            loss = loss / distributed.get_world_size()
            loss.backward()
        distributed.sync_grad_sum(run_manager.net)
        run_manager.optimizer.step()
        losses.update(list_mean(loss_of_subnets), images.size(0))
        top1.update(list_mean(acc1_of_subnets), images.size(0))
        top5.update(list_mean(acc5_of_subnets), images.size(0))

        if i % 100 == 0 and torch.distributed.get_rank() == 0:
            string = f"Epoch [{epoch}] Iter [{i}/{nBatch}] "
            for key, value in {
                'task': args.task,
                'phase': args.phase,
                'loss': losses.avg.item(),
                'top1': top1.avg.item(),
                'top5': top5.avg.item(),
                'R': images.size(2),
                'lr': new_lr,
                'loss_type': loss_type,
                'seed': str(subnet_seed),
                'str': subnet_str,
                'data_time': data_time.avg,
            }.items():
                string += f"{key}: {value}, "
            print(string)

        # t.set_postfix({
        #     'task':args.task,
        #     'phase':args.phase,
        #     'loss': losses.avg.item(),
        #     'top1': top1.avg.item(),
        #     'top5': top5.avg.item(),
        #     'R': images.size(2),
        #     'lr': new_lr,
        #     'loss_type': loss_type,
        #     'seed': str(subnet_seed),
        #     'str': subnet_str,
        #     'data_time': data_time.avg,
        # })
        # t.update(1)
        end = time.time()

    # with tqdm(total=nBatch,
    #           desc='Train Epoch #{}'.format(epoch + 1),
    #           disable=not run_manager.is_root) as t:
    #     end = time.time()
    #     for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
    #         data_time.update(time.time() - end)
    #         if epoch < warmup_epochs:
    #             new_lr = run_manager.run_config.warmup_adjust_learning_rate(
    #                 run_manager.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
    #             )
    #         else:
    #             new_lr = run_manager.run_config.adjust_learning_rate(
    #                 run_manager.optimizer, epoch - warmup_epochs, i, nBatch
    #             )
    #
    #         images, labels = images.cuda(), labels.cuda()
    #         target = labels
    #
    #         # soft target
    #         if args.kd_ratio > 0:
    #             args.teacher_model.train()
    #             with torch.no_grad():
    #                 soft_logits = args.teacher_model(images).detach()
    #                 soft_label = F.softmax(soft_logits, dim=1)
    #
    #         # clear gradients
    #         run_manager.optimizer.zero_grad()
    #
    #         loss_of_subnets, acc1_of_subnets, acc5_of_subnets = [], [], []
    #         # compute output
    #         subnet_str = ''
    #         for _ in range(args.dynamic_batch_size):
    #
    #             # set random seed before sampling
    #             if args.independent_distributed_sampling:
    #                 subnet_seed = os.getpid() + time.time()
    #             else:
    #                 subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, _, 0))
    #             random.seed(subnet_seed)
    #             subnet_settings = dynamic_net.sample_active_subnet()
    #             subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
    #                 key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
    #             ) for key, val in subnet_settings.items()]) + ' || '
    #
    #             output = run_manager.net(images)
    #             if args.kd_ratio == 0:
    #                 loss = run_manager.train_criterion(output, labels)
    #                 loss_type = 'ce'
    #             else:
    #                 if args.kd_type == 'ce':
    #                     kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
    #                 else:
    #                     kd_loss = F.mse_loss(output, soft_logits)
    #                 loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
    #                 loss = loss * (2 / (args.kd_ratio + 1))
    #                 loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)
    #
    #             # measure accuracy and record loss
    #             acc1, acc5 = accuracy(output, target, topk=(1, 5))
    #             loss_of_subnets.append(loss)
    #             acc1_of_subnets.append(acc1[0])
    #             acc5_of_subnets.append(acc5[0])
    #             loss = loss / distributed.get_world_size()
    #             loss.backward()
    #         distributed.sync_grad_sum(run_manager.net)
    #         run_manager.optimizer.step()
    #         losses.update(list_mean(loss_of_subnets), images.size(0))
    #         top1.update(list_mean(acc1_of_subnets), images.size(0))
    #         top5.update(list_mean(acc5_of_subnets), images.size(0))
    #
    #         t.set_postfix({
    #             'task':args.task,
    #             'phase':args.phase,
    #             'loss': losses.avg.item(),
    #             'top1': top1.avg.item(),
    #             'top5': top5.avg.item(),
    #             'R': images.size(2),
    #             'lr': new_lr,
    #             'loss_type': loss_type,
    #             'seed': str(subnet_seed),
    #             'str': subnet_str,
    #             'data_time': data_time.avg,
    #         })
    #         t.update(1)
    #         end = time.time()
    return losses.avg.item(), top1.avg.item(), top5.avg.item()


def train(run_manager, args, validate_func=None):
    if validate_func is None:
        validate_func = validate

    for epoch in range(run_manager.start_epoch, run_manager.run_config.n_epochs + args.warmup_epochs):
        train_loss, train_top1, train_top5 = train_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr)

        if (epoch + 1) % args.validation_frequency == 0:
            # validate under train mode
            val_loss, val_acc, val_acc5, _val_log = validate_func(run_manager, epoch=epoch, is_test=True)
            # best_acc
            is_best = val_acc > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, val_acc)
            if run_manager.is_root:
                val_log = 'Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - args.warmup_epochs, run_manager.run_config.n_epochs, val_loss, val_acc,
                           run_manager.best_acc)
                val_log += ', Train top-1 {top1:.3f}, Train loss {loss:.3f}\t'.format(top1=train_top1, loss=train_loss)
                val_log += _val_log
                run_manager.write_log(val_log, 'valid', should_print=False)

                run_manager.save_model({
                    'epoch': epoch,
                    'best_acc': run_manager.best_acc,
                    'optimizer': run_manager.optimizer.state_dict(),
                    'state_dict': run_manager.net.state_dict(),
                }, is_best=is_best)


def load_models(run_manager, dynamic_net, model_path=None):
    # specify init path
    init = torch.load(model_path, map_location='cpu')['state_dict']
    dynamic_net.load_weights_from_net(init)
    run_manager.write_log('Loaded init from %s' % model_path, 'valid')


def supporting_elastic_depth(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    # load stage info
    stage_info_path = os.path.join(run_manager.path, 'depth.stage')
    try:
        stage_info = json.load(open(stage_info_path))
    except Exception:
        stage_info = {'stage': 0}

    # load pretrained models
    validate_func_dict['depth_list'] = sorted(dynamic_net.depth_list)

    if args.phase == 1:
        load_models(run_manager, dynamic_net, model_path='exp/normal2kernel/checkpoint/model_best.pth.tar')
    else:
        load_models(run_manager, dynamic_net, model_path='exp/kernel2kernel_depth/phase1/checkpoint/model_best.pth.tar')

    # validate after loading weights
    run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')

    depth_stage_list = dynamic_net.depth_list.copy()
    depth_stage_list.sort(reverse=True)
    n_stages = len(depth_stage_list) - 1
    start_stage = n_stages - 1
    for current_stage in range(start_stage, n_stages):
        run_manager.write_log(
            '-' * 30 + 'Supporting Elastic Depth: %s -> %s' %
            (depth_stage_list[:current_stage + 1], depth_stage_list[:current_stage + 2]) + '-' * 30, 'valid'
        )

        # add depth list constraints
        supported_depth = depth_stage_list[:current_stage + 2]
        if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.expand_ratio_list)) == 1:
            validate_func_dict['depth_list'] = supported_depth
        else:
            validate_func_dict['depth_list'] = sorted({min(supported_depth), max(supported_depth)})
        dynamic_net.set_constraint(supported_depth, constraint_type='depth')

        # train
        train_func(
            run_manager, args,
            lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
        )

        # next stage & reset
        stage_info['stage'] += 1
        run_manager.start_epoch = 0
        run_manager.best_acc = 0.0

        # save and validate
        run_manager.save_model(model_name='depth_stage%d.pth.tar' % stage_info['stage'])
        json.dump(stage_info, open(stage_info_path, 'w'), indent=4)
        validate_func_dict['depth_list'] = sorted(dynamic_net.depth_list)
        run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')


def supporting_elastic_expand(train_func, run_manager, args, validate_func_dict):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    # load stage info
    stage_info_path = os.path.join(run_manager.path, 'expand.stage')
    try:
        stage_info = json.load(open(stage_info_path))
    except Exception:
        stage_info = {'stage': 0}

    # load pretrained models
    validate_func_dict['expand_ratio_list'] = sorted(dynamic_net.expand_ratio_list)

    if args.phase == 1:
        load_models(run_manager, dynamic_net, model_path='exp/kernel2kernel_depth/phase2/checkpoint/model_best.pth.tar')
    else:
        load_models(run_manager, dynamic_net,
                    model_path='exp/kernel_depth2kernel_depth_width/phase1/checkpoint/model_best.pth.tar')
    dynamic_net.re_organize_middle_weights()
    run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')

    expand_stage_list = dynamic_net.expand_ratio_list.copy()
    expand_stage_list.sort(reverse=True)
    n_stages = len(expand_stage_list) - 1
    start_stage = n_stages - 1

    for current_stage in range(start_stage, n_stages):
        run_manager.write_log(
            '-' * 30 + 'Supporting Elastic Expand Ratio: %s -> %s' %
            (expand_stage_list[:current_stage + 1], expand_stage_list[:current_stage + 2]) + '-' * 30, 'valid'
        )

        # add expand list constraints
        supported_expand = expand_stage_list[:current_stage + 2]
        if len(set(dynamic_net.ks_list)) == 1 and len(set(dynamic_net.depth_list)) == 1:
            validate_func_dict['expand_ratio_list'] = supported_expand
        else:
            validate_func_dict['expand_ratio_list'] = sorted({min(supported_expand), max(supported_expand)})
        dynamic_net.set_constraint(supported_expand, constraint_type='expand_ratio')

        # train
        train_func(
            run_manager, args,
            lambda _run_manager, epoch, is_test: validate(_run_manager, epoch, is_test, **validate_func_dict)
        )

        # next stage & reset
        stage_info['stage'] += 1
        run_manager.start_epoch = 0
        run_manager.best_acc = 0.0
        dynamic_net.re_organize_middle_weights(expand_ratio_stage=stage_info['stage'])
        if isinstance(run_manager, DistributedRunManager):
            run_manager.broadcast()

        # save and validate
        run_manager.save_model(model_name='expand_stage%d.pth.tar' % stage_info['stage'])
        json.dump(stage_info, open(stage_info_path, 'w'), indent=4)
        validate_func_dict['expand_ratio_list'] = sorted(dynamic_net.expand_ratio_list)
        run_manager.write_log('%.3f\t%.3f\t%.3f\t%s' % validate(run_manager, **validate_func_dict), 'valid')


def train_supernet(run_manager, args, validate_func=None, cur_model_id=-1):
    if validate_func is None:
        validate_func = validate

    for epoch in range(run_manager.start_epoch, run_manager.end_epoch + args.warmup_epochs):
        train_loss, train_top1, train_top5 = train_supernet_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr)

        if (epoch + 1) % args.validation_frequency == 0:
            # validate under train mode
            val_loss, val_acc, val_acc5, _val_log = validate_func(run_manager, epoch=epoch, is_test=True)
            # best_acc
            is_best = val_acc > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, val_acc)
            if run_manager.is_root:
                val_log = 'Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - args.warmup_epochs, run_manager.run_config.n_epochs, val_loss, val_acc,
                           run_manager.best_acc)
                val_log += ', Train top-1 {top1:.3f}, Train loss {loss:.3f}\t'.format(top1=train_top1, loss=train_loss)
                val_log += _val_log
                run_manager.write_log(val_log, 'valid', should_print=False)

                if cur_model_id >= 0:
                    run_manager.save_model({
                        'epoch': epoch,
                        'best_acc': run_manager.best_acc,
                        'optimizer': run_manager.optimizer.state_dict(),
                        'state_dict': run_manager.net.state_dict(),
                    }, is_best=is_best, model_name='submodel_' + str(cur_model_id) + '_checkpoint.pth.tar',
                        cur_model_id=cur_model_id)
                else:
                    run_manager.save_model({
                        'epoch': epoch,
                        'best_acc': run_manager.best_acc,
                        'optimizer': run_manager.optimizer.state_dict(),
                        'state_dict': run_manager.net.state_dict(),
                    }, is_best=is_best)


# different submodels are sampled during each step, many submodels are trained in one epoch
# total num of epochs = warmup_epochs + num_epoch
# dynamic_batchsize sets the number of different models trained in each step(using the same data), loss is then backpropped together
def train_supernet_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0):
    dynamic_net = run_manager.net

    # switch to train mode
    dynamic_net.train()
    run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric('train_loss')
    top1 = DistributedMetric('train_top1')
    top5 = DistributedMetric('train_top5')

    # with tqdm(total=nBatch,
    #           desc='Train Epoch #{}'.format(epoch + 1),
    #           disable=not run_manager.is_root) as t:
    end = time.time()
    for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
        data_time.update(time.time() - end)
        if epoch < warmup_epochs:
            new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                run_manager.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
            )
        else:
            new_lr = run_manager.run_config.adjust_learning_rate(
                run_manager.optimizer, epoch - warmup_epochs, i, nBatch
            )

        images, labels = images.cuda(), labels.cuda()
        target = labels

        # soft target
        if args.kd_ratio > 0:
            args.teacher_model.train()
            with torch.no_grad():
                soft_logits = args.teacher_model(images).detach()
                soft_label = F.softmax(soft_logits, dim=1)

        # clear gradients
        run_manager.optimizer.zero_grad()

        loss_of_subnets, acc1_of_subnets, acc5_of_subnets = [], [], []
        # compute output
        subnet_str = ''
        for _ in range(args.dynamic_batch_size):

            # set random seed before sampling
            if args.independent_distributed_sampling:
                subnet_seed = os.getpid() + time.time()
            else:
                subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, _, 0))
            random.seed(subnet_seed)
            subnet_settings = dynamic_net.sample_active_subnet()
            subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
            ) for key, val in subnet_settings.items()]) + ' || '

            output = run_manager.net(images)
            if args.kd_ratio == 0:
                loss = run_manager.train_criterion(output, labels)
                loss_type = 'ce'
            else:
                if args.kd_type == 'ce':
                    kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                else:
                    kd_loss = F.mse_loss(output, soft_logits)
                loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
                loss = loss * (2 / (args.kd_ratio + 1))
                loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss_of_subnets.append(loss)
            acc1_of_subnets.append(acc1[0])
            acc5_of_subnets.append(acc5[0])
            loss = loss / distributed.get_world_size()
            loss.backward()
        distributed.sync_grad_sum(run_manager.net)
        run_manager.optimizer.step()
        losses.update(list_mean(loss_of_subnets), images.size(0))
        top1.update(list_mean(acc1_of_subnets), images.size(0))
        top5.update(list_mean(acc5_of_subnets), images.size(0))

        if i % 100 == 0 and torch.distributed.get_rank() == 0:

            string = f"Train Epoch [{epoch}] Iter [{i}/{nBatch}] "
            for key, value in {
                'task': args.task,
                'phase': args.phase,
                'loss': "{:.3f}".format(losses.avg.item()),
                'top1': "{:.3f}".format(top1.avg.item()),
                'top5': "{:.3f}".format(top5.avg.item()),
                'R': images.size(2),
                'lr': "{:.3f}".format(new_lr),
                'loss_type': loss_type,
                'seed': str(subnet_seed),
                'str': subnet_str,
                'data_time': "{:.3f}".format(data_time.avg),
            }.items():
                string += f"{key}: {value}, "
            print(string)
        # if i >= 100:
        #     break
        # args.logging.info(string)
        # t.set_postfix({
        #     'task':args.task,
        #     'phase':args.phase,
        #     'loss': losses.avg.item(),
        #     'top1': top1.avg.item(),
        #     'top5': top5.avg.item(),
        #     'R': images.size(2),
        #     'lr': new_lr,
        #     'loss_type': loss_type,
        #     'seed': str(subnet_seed),
        #     'str': subnet_str,
        #     'data_time': data_time.avg,
        # })
        # t.update(1)
        end = time.time()
    return losses.avg.item(), top1.avg.item(), top5.avg.item()


def validate_supernet(run_manager, epoch=0, is_test=True, image_size_list=None,
                      width_mult_list=None, ks_list=None, expand_ratio_list=None, depth_list=None,
                      additional_setting=None):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    if image_size_list is None:
        image_size_list = int2list(run_manager.run_config.data_provider.image_size, 1)
    if width_mult_list is None:
        width_mult_list = [i for i in range(len(dynamic_net.width_mult_list))]
    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list

    subnet_settings = []
    for w in width_mult_list:
        for d in depth_list:
            for e in expand_ratio_list:
                for k in ks_list:
                    for img_size in image_size_list:
                        subnet_settings.append([{
                            'image_size': img_size,
                            'wid': w,
                            'd': d,
                            'e': e,
                            'ks': k,
                        }, 'R%s-W%s-D%s-E%s-K%s' % (img_size, w, d, e, k)])
    # for w in width_mult_list:
    #     for d in depth_list:
    #         for e in expand_ratio_list:
    #             for k in ks_list:
    #                 for img_size in image_size_list:
    #                     subnet_settings.append([{
    #                         'image_size': img_size,
    #                         'wid': w,
    #                         'd': d,
    #                         'e': e,
    #                         'ks': k,
    #                     }, 'R%s-W%s-D%s-E%s-K%s' % (img_size, w, d, e, k)])
    if additional_setting is not None:
        subnet_settings += additional_setting

    losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []

    valid_log = ''
    for setting, name in subnet_settings:
        run_manager.write_log('-' * 30 + ' Validate %s ' % name + '-' * 30, 'train', should_print=True)
        run_manager.run_config.data_provider.assign_active_img_size(setting.pop('image_size'))
        # setting.pop('image_size')
        dynamic_net.set_active_subnet(**setting)
        run_manager.write_log(dynamic_net.module_str, 'train', should_print=True)

        run_manager.reset_running_statistics(dynamic_net)
        loss, top1, top5 = run_manager.validate(epoch=epoch, is_test=is_test, run_str=name, net=dynamic_net)
        losses_of_subnets.append(loss)
        top1_of_subnets.append(top1)
        top5_of_subnets.append(top5)
        valid_log += '%s (%.3f), ' % (name, top1)

    return list_mean(losses_of_subnets), list_mean(top1_of_subnets), list_mean(top5_of_subnets), valid_log


def match_loss(gw_syn, gw_real, dis_metric='per-filter-cos'):
    dis = torch.tensor(0.0).cuda()

    if dis_metric == 'per-filter-cos':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                    torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis


def mincut_split_ofa(dist_avg, split_num):  # note: this is not strictly mincut, but it's fine for this
    assert (
                split_num == 2 or split_num == 3), 'always split into 2 or 3 groups for darts space (when using gradient to split)'
    assert isinstance(dist_avg, np.ndarray)

    vertex = [i for i in range(dist_avg.shape[0])]

    max_cut = 100000
    if split_num == 2:
        for subset in chain(*map(lambda x: combinations(vertex, x), range(1, len(vertex) + 1))):
            if len(subset) >= 1 and len(subset) <= len(vertex) // 2:
                cut = 0
                for edge in combinations(vertex, 2):
                    if (edge[0] in subset and edge[1] in subset):
                        cut += dist_avg[edge[0], edge[1]]
                    if (edge[0] not in subset and edge[1] not in subset):
                        cut += dist_avg[edge[0], edge[1]]
                if cut < max_cut:
                    group0 = np.array([i for i in vertex if i in subset])
                    group1 = np.array([i for i in vertex if i not in subset])
                    # max_cut = cut
                    max_cut = (dist_avg - np.tril(dist_avg)).sum() - cut
        best_groups = [group0, group1]

    elif split_num == 3:
        group1 = np.array([0])
        group2 = np.array([1])
        group3 = np.array([2])

        max_cut = (dist_avg - np.tril(dist_avg)).sum()
        best_groups = [group1, group2, group3]

    return best_groups, max_cut


def random_split_ofa(split_num, num_ops):  # when split_num == num_ops -> split every operation like few-shot NAS
    assert num_ops % split_num == 0, 'always split into even groups for 201'
    if split_num == num_ops:  # exhaustive split
        opids = np.arange(0, num_ops)
    else:
        opids = np.random.permutation(num_ops)
    group_size = num_ops // split_num
    groups = [opids[s:s + group_size] for s in np.arange(0, num_ops, group_size)]

    return groups


# split a supernet into subnets, return encodings of splitted supernet
def split_supernet(run_manager, args, split_eid, split_crit, split_num, dis_metric='cos'):
    # switch to train mode
    run_manager.net.train()
    if split_crit == 'grad':
        if split_eid is None:
            eids = []
            # for i in range(1, 3):
            for i in range(1, len(run_manager.net.blocks)):
                if run_manager.net.blocks[i].mobile_inverted_conv.kernel_size_enc.sum(dim=-1) == run_manager.net.blocks[
                    i].mobile_inverted_conv.kernel_size_enc.size(0):
                    eids.append(i)
        else:
            eids = [split_eid]
        best_edge_score, best_eid, best_groups = 0, 9999, None
        for eid in eids:
            repeat = 100
            dist_avg = 0
            # print(eid, run_manager.net.blocks[eid].mobile_inverted_conv.kernel_size_enc)
            n_choices = run_manager.net.blocks[eid].mobile_inverted_conv.kernel_size_enc.size(0)

            for _ in range(repeat):
                encs = [None]
                ## fetch data (one batch for now)
                images, labels = next(iter(run_manager.run_config.train_loader))
                images, labels = images.cuda(), labels.cuda()
                target = labels

                # soft target
                if args.kd_ratio > 0:
                    args.teacher_model.train()
                with torch.no_grad():
                    soft_logits = args.teacher_model(images).detach()
                soft_label = F.softmax(soft_logits, dim=1)

                # clear gradients
                run_manager.optimizer.zero_grad()
                subnet_settings = run_manager.net.sample_active_subnet()

                split_op_grads = []
                for opid in range(n_choices):
                    run_manager.net.blocks[eid].mobile_inverted_conv.active_kernel_size = \
                    run_manager.net.blocks[eid].mobile_inverted_conv.kernel_size_list[opid]

                    output = run_manager.net(images)
                    if args.kd_ratio == 0:
                        loss = run_manager.train_criterion(output, labels)
                        loss_type = 'ce'
                    else:
                        if args.kd_type == 'ce':
                            kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                        else:
                            kd_loss = F.mse_loss(output, soft_logits)
                        loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
                        loss = loss * (2 / (args.kd_ratio + 1))

                    loss = loss / distributed.get_world_size()
                    run_manager.net.zero_grad()
                    loss.backward()
                    distributed.sync_grad_sum(run_manager.net)

                    ## get gradients
                    grads = run_manager.net.get_split_gradients(split_eid=eid)
                    grads = [g.clone().detach() for g in grads]
                    split_op_grads.append(grads)

                ## compute matching scores (redundant as dist_mat is symmetric, but good for debugging)
                dist_mat = torch.zeros((n_choices, n_choices))
                for opid_i in range(n_choices):
                    for opid_j in range(n_choices):
                        dist_mat[opid_i, opid_j] = match_loss(split_op_grads[opid_i], split_op_grads[opid_j],
                                                              dis_metric=dis_metric)
                dist_avg += dist_mat
            dist_avg /= repeat
            if run_manager.is_root:
                print('\n' + 'edge ' + str(eid) + ' distance matrix:')
                print('\n' + str(dist_avg))  # TODO: write in the Writer

            ## partition
            groups, edge_score = mincut_split_ofa(dist_avg.numpy(),
                                                  split_num)  # TODO implement the max-cut algorithm to split the supernet
            if run_manager.is_root:
                print('edge ' + str(eid), groups, edge_score)

            ## compute edge score
            if edge_score > best_edge_score:
                best_edge_score = edge_score
                best_eid = eid
                best_groups = groups

        split_eid = best_eid
        groups = best_groups

    elif split_crit == 'fewshot':  # when num_ops == split_num, reuse random split
        eid = split_eid
        n_choices = run_manager.net.blocks[eid].mobile_inverted_conv.kernel_size_enc.size(0)
        groups = random_split_ofa(split_num, n_choices)
    else:
        print(f"ERROR: UNRECOGNIZED SPLIT CRITERIA: {split_crit}");
        exit(1)

    encs_splitted = []
    for group in groups:
        n_choices = run_manager.net.blocks[eid].mobile_inverted_conv.kernel_size_enc.size(0)
        enc = torch.zeros(n_choices)
        enc[torch.LongTensor(group)] = 1
        encs_splitted.append(enc)
    return encs_splitted, split_eid


def finetune_validate(run_manager, epoch=0, is_test=True, subnet_settings=None, image_size_list=None,
                      width_mult_list=None, ks_list=None, expand_ratio_list=None, depth_list=None,
                      additional_setting=None):
    dynamic_net = run_manager.net
    if isinstance(dynamic_net, nn.DataParallel):
        dynamic_net = dynamic_net.module

    dynamic_net.eval()

    if image_size_list is None:
        image_size_list = int2list(run_manager.run_config.data_provider.image_size, 1)
    if width_mult_list is None:
        width_mult_list = [i for i in range(len(dynamic_net.width_mult_list))]
    if ks_list is None:
        ks_list = dynamic_net.ks_list
    if expand_ratio_list is None:
        expand_ratio_list = dynamic_net.expand_ratio_list
    if depth_list is None:
        depth_list = dynamic_net.depth_list

    # subnet_settings = []
    # for w in width_mult_list:
    #     for d in depth_list:
    #         for e in expand_ratio_list:
    #             for k in ks_list:
    #                 for img_size in image_size_list:
    #                     subnet_settings.append([{
    #                         'image_size': img_size,
    #                         'wid': w,
    #                         'd': d,
    #                         'e': e,
    #                         'ks': k,
    #                     }, 'R%s-W%s-D%s-E%s-K%s' % (img_size, w, d, e, k)])
    # if additional_setting is not None:
    #     subnet_settings += additional_setting

    losses_of_subnets, top1_of_subnets, top5_of_subnets = [], [], []

    valid_log = ''
    run_manager.write_log('-' * 30 + ' Validate ' + '-' * 30, 'train', should_print=True)
    run_manager.run_config.data_provider.assign_active_img_size(224)
    # setting.pop('image_size')
    # dynamic_net.set_active_subnet(**subnet_settings)
    dynamic_net.set_active_subnet(ks=subnet_settings['ks'], e=subnet_settings['e'],
                                  d=subnet_settings['d'])
    run_manager.write_log(dynamic_net.module_str, 'train', should_print=True)

    run_manager.reset_running_statistics(dynamic_net)
    loss, top1, top5 = run_manager.validate(epoch=epoch, is_test=is_test, net=dynamic_net)
    losses_of_subnets.append(loss)
    top1_of_subnets.append(top1)
    top5_of_subnets.append(top5)
    valid_log += '(%.3f), ' % (top1)
    # for setting, name in subnet_settings:
    #     print(setting)
    #     run_manager.write_log('-' * 30 + ' Validate %s ' % name + '-' * 30, 'train', should_print=True)
    #     run_manager.run_config.data_provider.assign_active_img_size(setting.pop('image_size'))
    #     # setting.pop('image_size')
    #     dynamic_net.set_active_subnet(**setting)
    #     run_manager.write_log(dynamic_net.module_str, 'train', should_print=True)
    #
    #     run_manager.reset_running_statistics(dynamic_net)
    #     loss, top1, top5 = run_manager.validate(epoch=epoch, is_test=is_test, run_str=name, net=dynamic_net)
    #     losses_of_subnets.append(loss)
    #     top1_of_subnets.append(top1)
    #     top5_of_subnets.append(top5)
    #     valid_log += '%s (%.3f), ' % (name, top1)

    return list_mean(losses_of_subnets), list_mean(top1_of_subnets), list_mean(top5_of_subnets), valid_log


def finetune_net(run_manager, args, subnet_settings=None, validate_func=None):
    if validate_func is None:
        validate_func = finetune_validate

    for epoch in range(run_manager.start_epoch, run_manager.end_epoch + args.warmup_epochs):
        train_loss, train_top1, train_top5 = finetune_net_one_epoch(
            run_manager, args, epoch, args.warmup_epochs, args.warmup_lr, subnet_settings)

        if (epoch + 1) % args.validation_frequency == 0:
            # validate under train mode
            val_loss, val_acc, val_acc5, _val_log = validate_func(run_manager, epoch=epoch, is_test=True,
                                                                  subnet_settings=subnet_settings)
            # best_acc
            is_best = val_acc > run_manager.best_acc
            run_manager.best_acc = max(run_manager.best_acc, val_acc)
            if run_manager.is_root:
                val_log = 'Valid [{0}/{1}] loss={2:.3f}, top-1={3:.3f} ({4:.3f})'. \
                    format(epoch + 1 - args.warmup_epochs, run_manager.run_config.n_epochs, val_loss, val_acc,
                           run_manager.best_acc)
                val_log += ', Train top-1 {top1:.3f}, Train loss {loss:.3f}\t'.format(top1=train_top1, loss=train_loss)
                val_log += _val_log
                run_manager.write_log(val_log, 'valid', should_print=False)

                run_manager.save_model({
                    'epoch': epoch,
                    'best_acc': run_manager.best_acc,
                    'optimizer': run_manager.optimizer.state_dict(),
                    'state_dict': run_manager.net.state_dict(),
                }, is_best=is_best)


def finetune_net_one_epoch(run_manager, args, epoch, warmup_epochs=0, warmup_lr=0, subnet_settings=None):
    dynamic_net = run_manager.net

    # switch to train mode
    dynamic_net.train()
    run_manager.run_config.train_loader.sampler.set_epoch(epoch)
    MyRandomResizedCrop.EPOCH = epoch

    nBatch = len(run_manager.run_config.train_loader)

    data_time = AverageMeter()
    losses = DistributedMetric('train_loss')
    top1 = DistributedMetric('train_top1')
    top5 = DistributedMetric('train_top5')

    end = time.time()
    for i, (images, labels) in enumerate(run_manager.run_config.train_loader):
        data_time.update(time.time() - end)
        if epoch < warmup_epochs:
            new_lr = run_manager.run_config.warmup_adjust_learning_rate(
                run_manager.optimizer, warmup_epochs * nBatch, nBatch, epoch, i, warmup_lr,
            )
        else:
            new_lr = run_manager.run_config.adjust_learning_rate(
                run_manager.optimizer, epoch - warmup_epochs, i, nBatch
            )

        images, labels = images.cuda(), labels.cuda()
        target = labels

        if isinstance(run_manager.run_config.mixup_alpha, float):
            # transform data
            random.seed(int('%d%.3d' % (i, epoch)))
            lam = random.betavariate(run_manager.run_config.mixup_alpha, run_manager.run_config.mixup_alpha)
            images = mix_images(images, lam)
            labels = mix_labels(
                labels, lam, run_manager.run_config.data_provider.n_classes, run_manager.run_config.label_smoothing
            )

        # soft target
        if args.kd_ratio > 0:
            args.teacher_model.train()
            with torch.no_grad():
                soft_logits = args.teacher_model(images).detach()
                soft_label = F.softmax(soft_logits, dim=1)

        # clear gradients
        run_manager.optimizer.zero_grad()

        loss_of_subnets, acc1_of_subnets, acc5_of_subnets = [], [], []
        # compute output
        subnet_str = ''
        for _ in range(args.dynamic_batch_size):

            # set random seed before sampling
            if args.independent_distributed_sampling:
                subnet_seed = os.getpid() + time.time()
            else:
                subnet_seed = int('%d%.3d%.3d' % (epoch * nBatch + i, _, 0))
            random.seed(subnet_seed)
            # subnet_settings = dynamic_net.sample_active_subnet()
            dynamic_net.set_active_subnet(ks=subnet_settings['ks'], e=subnet_settings['e'],
                                          d=subnet_settings['d'])
            subnet_str += '%d: ' % _ + ','.join(['%s_%s' % (
                key, '%.1f' % subset_mean(val, 0) if isinstance(val, list) else val
            ) for key, val in subnet_settings.items()]) + ' || '

            output = run_manager.net(images)
            if args.kd_ratio == 0:
                loss = run_manager.train_criterion(output, labels)
                loss_type = 'ce'
            else:
                if args.kd_type == 'ce':
                    kd_loss = cross_entropy_loss_with_soft_target(output, soft_label)
                else:
                    kd_loss = F.mse_loss(output, soft_logits)
                loss = args.kd_ratio * kd_loss + run_manager.train_criterion(output, labels)
                loss = loss * (2 / (args.kd_ratio + 1))
                loss_type = '%.1fkd-%s & ce' % (args.kd_ratio, args.kd_type)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss_of_subnets.append(loss)
            acc1_of_subnets.append(acc1[0])
            acc5_of_subnets.append(acc5[0])
            loss = loss / distributed.get_world_size()
            loss.backward()

        distributed.sync_grad_sum(run_manager.net)
        run_manager.optimizer.step()
        losses.update(list_mean(loss_of_subnets), images.size(0))
        top1.update(list_mean(acc1_of_subnets), images.size(0))
        top5.update(list_mean(acc5_of_subnets), images.size(0))

        if i % 100 == 0 and torch.distributed.get_rank() == 0:

            string = f"Train Epoch [{epoch}] Iter [{i}/{nBatch}] "
            for key, value in {
                'task': args.task,
                'phase': args.phase,
                'loss': "{:.3f}".format(losses.avg.item()),
                'top1': "{:.3f}".format(top1.avg.item()),
                'top5': "{:.3f}".format(top5.avg.item()),
                'R': images.size(2),
                'lr': "{:.3f}".format(new_lr),
                'loss_type': loss_type,
                'seed': str(subnet_seed),
                'data_time': "{:.3f}".format(data_time.avg),
            }.items():
                string += f"{key}: {value}, "
            print(string)

        end = time.time()
    return losses.avg.item(), top1.avg.item(), top5.avg.item()



