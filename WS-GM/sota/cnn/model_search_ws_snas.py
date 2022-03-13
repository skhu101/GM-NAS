import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from sota.cnn.operations import *
from sota.cnn.genotypes import Genotype
import sys
sys.path.insert(0, '../../')
#from nasbench201.utils import drop_path
from sota.cnn.model_search import Network, Cell
from sota.cnn.model_search import Cell as node_str_dict

class NetworkSNASWS(Network):
    def __init__(self, C, num_classes, layers, criterion, primitives, args,
                 steps=4, multiplier=4, stem_multiplier=3, drop_path_prob=0):
      super(NetworkSNASWS, self).__init__(C, num_classes, layers, criterion, primitives, args, steps=steps, multiplier=multiplier, stem_multiplier=stem_multiplier, drop_path_prob=drop_path_prob)

      self.weights_map = lambda x: torch.softmax(x, dim=-1)
      self._init_encoding()
      self.mode = args.supernet_train
      self.space_name = 'darts'
      self.tau = args.tau_max

    def _init_encoding(self):
      ## TODO under dev ##
      self.enc_normal = torch.ones_like(self._arch_parameters[0]).requires_grad_(False) # gpu
      self.enc_reduce = torch.ones_like(self._arch_parameters[1]).requires_grad_(False) # gpu

    def set_encoding(self, enc_normal, enc_reduce):
      self.enc_normal.data.copy_(enc_normal.data)
      self.enc_reduce.data.copy_(enc_reduce.data)

    def get_depth(self): # how many splits (depth of this supernet in the split tree)
      depth = 0 
      for enc_eid in self.enc_normal:
        if 0 in enc_eid: depth += 1
      for enc_eid in self.enc_reduce:
        if 0 in enc_eid: depth += 1
      return depth

    def _check_connect(self, discret_weights): # must sample valid connections (required if none op exists)
        return True
    #  none_id = 0 
    #  if discret_weights[3, none_id] != 1: return True
    #  if discret_weights[1, none_id] != 1 and discret_weights[5, none_id] != 1: return True
    #  if discret_weights[0, none_id] != 1 and discret_weights[4, none_id] != 1: return True
    #  if discret_weights[0, none_id] != 1 and discret_weights[2, none_id] != 1 and discret_weights[5, none_id] != 1: return True

    def _get_gumbel_softmax_weights(self):
        # weights_normal
        weights_normal = self._arch_parameters[0] * self.enc_normal
        for eid in range(self.enc_normal.shape[0]):
          weights_normal[eid][self.enc_normal[eid] == 1] = torch.nn.functional.gumbel_softmax(weights_normal[eid][self.enc_normal[eid] == 1], tau=self.tau, dim=-1)
        # weights_reduce
        weights_reduce = self._arch_parameters[1] * self.enc_reduce
        for eid in range(self.enc_reduce.shape[0]):
          weights_reduce[eid][self.enc_reduce[eid] == 1] = torch.nn.functional.gumbel_softmax(weights_reduce[eid][self.enc_reduce[eid] == 1], tau=self.tau, dim=-1)
        return weights_normal, weights_reduce

    def get_split_gradients(self, split_eid=0, split_normal=True): # get all gradients except for edge [eid]
      params = []
      for i, cell in enumerate(self.cells):
        if isinstance(cell, Cell):
          if cell.reduction and not split_normal:
            for eid in range(len(cell._ops)):
                if eid != split_eid-self._arch_parameters[0].shape[0]:
                    params += list(cell._ops[eid].parameters())
          elif not cell.reduction and split_normal:
            for eid in range(len(cell._ops)):
                if eid != split_eid:
                    params += list(cell._ops[eid].parameters())
          else:
            for eid in range(len(cell._ops)):
                params += list(cell._ops[eid].parameters())
        else:
          params += list(cell.parameters())
        #if isinstance(cell, Cell):
        #  if cell.reduction and not split_normal:
        #    for eid in range(len(cell.edges)):
        #        if eid != split_eid-self._arch_parameters[0].shape[0]:
        #            params += list(cell.edges[node_str_dict[eid]].parameters())
        #  elif not cell.reduction and split_normal:
        #    for eid in range(len(cell.edges)):
        #        if eid != split_eid:
        #            params += list(cell.edges[node_str_dict[eid]].parameters())
        #  else:
        #    for eid in range(len(cell.edges)):
        #        params += list(cell.edges[node_str_dict[eid]].parameters())
        #else:
        #  params += list(cell.parameters())
      param_grads = [p.grad for p in params if p.grad != None] #and p.grad.sum() != 0]
      return param_grads


    def get_theta(self):
      if self.mode == 'snas':
        return self._get_gumbel_softmax_weights()
      else:
        raise NotImplementedError

    def forward(self, input, weights_normal=None, weights_reduce=None):
        ## get weights
        if weights_normal is None and weights_reduce is None:
            weights_normal, weights_reduce = self.get_theta()
        #weights = self.get_softmax()
        #weights_normal = weights['normal']
        #weights_reduce = weights['reduce']

        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = weights_reduce
            else:
                weights = weights_normal

            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))

        return logits

    def genotype(self):
        def _parse(weights, normal=True):
            PRIMITIVES = self.PRIMITIVES['primitives_normal' if normal else 'primitives_reduct'] ## two are equal for Darts space

            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()

                try:
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES[x].index('none')))[:2]
                except ValueError: # This error happens when the 'none' op is not present in the ops
                    edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if 'none' in PRIMITIVES[j]:
                            if k != PRIMITIVES[j].index('none'):
                                if k_best is None or W[j][k] > W[j][k_best]:
                                    k_best = k
                        else:
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[start+j][k_best], j))
                start = end
                n += 1
            return gene

        weights_normal = self._arch_parameters[0] * self.enc_normal
        for eid in range(self.enc_normal.shape[0]):
          weights_normal[eid][self.enc_normal[eid] == 1] = torch.softmax(weights_normal[eid][self.enc_normal[eid] == 1], dim=-1)
          weights_normal[eid][self.enc_normal[eid] == 0] = 0
        
        weights_reduce = self._arch_parameters[1] * self.enc_reduce
        for eid in range(self.enc_reduce.shape[0]):
          weights_reduce[eid][self.enc_reduce[eid] == 1] = torch.softmax(weights_reduce[eid][self.enc_reduce[eid] == 1], dim=-1)
          weights_reduce[eid][self.enc_reduce[eid] == 0] = 0

        gene_normal = _parse(weights_normal.data.cpu().numpy(), True)
        gene_reduce = _parse(weights_reduce.data.cpu().numpy(), False)
        #gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy(), True)
        #gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy(), False)

        concat = range(2+self._steps-self._multiplier, self._steps+2)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    def new(self, enc_normal, enc_reduce): # for 2nd order
        #model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, self._args,\
        #                    drop_path_prob=self.drop_path_prob)
        # model_new = Network(self._C, self._num_classes, self._layers, self._criterion, self.PRIMITIVES, self._args,\
        #                     drop_path_prob=self.drop_path_prob).cuda()
        model_new = self.get_new_model() # get a new random model
        model_new.set_encoding(enc_normal, enc_reduce)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    ## this print function will output alpha * enc rather than alpha
    # def printing(self, logging, option='all'):
    #     weights_normal, weights_reduce = self.get_theta()
    #     if option in ['all', 'normal']:
    #         logging.info(weights_normal)
    #     if option in ['all', 'reduce']:
    #         logging.info(weights_reduce)