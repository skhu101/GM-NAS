import os
import sys
import numpy as np
import torch
sys.path.insert(0, '../')
import logging
import torch.utils

from nas_201_api import NASBench201API as API

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


def darts_project(train_queue, valid_queue, model, architect, criterion, optimizer,
                epoch, args, infer, query):
    ## TODO under dev ##
    if not args.fast or True:
        api = API('../data/NAS-Bench-201-v1_0-e61699.pth')

    logging.info('='*30)
    logging.info(model.genotype())
    model.printing(logging)
    
    valid_acc, valid_obj = infer(valid_queue, model, criterion, log=False, eval=('snas' in args.method))
    
    logging.info('valid_acc  %f', valid_acc)
    logging.info('valid_loss %f', valid_obj)
    if not args.fast:
        cur_result = query(api, model.genotype(), logging)
    alpha = model.arch_parameters()[0].clone()
    enc   = model.enc.clone()
    
    return valid_acc, alpha, enc


def rsws_project(train_queue, valid_queue, model, architect, criterion, optimizer,
                 epoch, args, infer, query):
    if not args.fast:
        api = API('../data/NAS-Bench-201-v1_0-e61699.pth')
    
    # randomly eval 200 architectures, and select the best one
    logging.info('='*30)
    best_valid_acc, best_valid_obj, best_theta, best_enc = 0, None, None, None
    repeat = 200 if not args.fast else 100
    sampled_archs = {}
    while len(sampled_archs) < repeat:
        model.train()
        # randomly sample an architecture (without replacement)
        theta = model.get_theta()
        arch_id = str(theta.argmax(dim=-1).cpu().numpy().tolist())
        if arch_id in sampled_archs:
            continue
        else:
            sampled_archs[arch_id] = 1

        valid_acc, valid_obj = infer(valid_queue, model, criterion, theta=theta, log=False, eval=True)
        
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_obj = valid_obj
            best_theta = theta
            best_enc   = model.enc.clone()

    ## log the best architecture from this supernet
    model.arch_parameters()[0].data.copy_(best_theta.data)
    logging.info('valid_acc  %f', best_valid_acc)
    logging.info('valid_loss %f', best_valid_obj)
    if not args.fast:
        query(api, model.genotype(), logging)

    return best_valid_acc, best_theta, best_enc
