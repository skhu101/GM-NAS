import torch
import torch.nn as nn
import numpy as np
from .search_cells import NAS201SearchCell as SearchCell, node_str_dict
from .search_model import TinyNetwork as TinyNetwork
from .genotypes        import Structure


class TinyNetworkWS(TinyNetwork):
  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space, args,
               affine=False, track_running_stats=True):
    super(TinyNetworkWS, self).__init__(C, N, max_nodes, num_classes, criterion, search_space, args,
          affine=affine, track_running_stats=track_running_stats)
    self.theta_map = lambda x: torch.softmax(x, dim=-1)
    self._init_encoding()
    self.mode = args.supernet_train
    self.space_name = '201'

  def _init_encoding(self):
    self.enc = torch.ones_like(self._arch_parameters).requires_grad_(False) # gpu

  def set_encoding(self, enc):
    self.enc.data.copy_(enc.data)
  
  def get_depth(self): # how many splits (depth of this supernet in the split tree)
    depth = 0
    for enc_eid in self.enc:
      if 0 in enc_eid: depth += 1
    return depth
  
  def get_unsplitted_eids(self):
    return torch.where(self.enc.sum(dim=-1) == self.enc.shape[1])[0]
  
  def check_connect(self, discrete_theta): # must sample valid connections (required if none op exists)
    if discrete_theta.shape[1] == 4: return True # nonone
    none_id = 0
    if discrete_theta[3, none_id] != 1: return True
    if discrete_theta[1, none_id] != 1 and discrete_theta[5, none_id] != 1: return True
    if discrete_theta[0, none_id] != 1 and discrete_theta[4, none_id] != 1: return True
    if discrete_theta[0, none_id] != 1 and discrete_theta[2, none_id] != 1 and discrete_theta[5, none_id] != 1: return True
    return False
    
  def _sample_single_path_theta(self): # randomly sample a single path
    while True:
      discrete_theta = torch.zeros_like(self._arch_parameters)
      for eid in range(self.enc.shape[0]):
        opids = torch.where(self.enc[eid].cpu() == 1)[0]
        selected_opid = np.random.choice(opids)
        discrete_theta[eid][selected_opid] = 1
      if self.check_connect(discrete_theta): break # changed Sep 9
    return discrete_theta

  def get_split_gradients(self, split_eid=0): # get all gradients except for edge [eid]
    params = []
    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        for eid in range(len(cell.edges)):
          if eid != split_eid:
            params += list(cell.edges[node_str_dict[eid]].parameters())
      else:
        params += list(cell.parameters())
    param_grads = [p.grad for p in params if p.grad != None]
    return param_grads
  
  def _get_softmax_theta(self):
      theta = self._arch_parameters * self.enc
      for eid in range(self.enc.shape[0]):
        theta[eid][self.enc[eid] == 1] = torch.softmax(theta[eid][self.enc[eid] == 1], dim=-1)
      return theta
      
  def get_theta(self):
    if   self.mode == 'rsws':
      return self._sample_single_path_theta()
    elif self.mode == 'darts':
      return self._get_softmax_theta()
    else:
      raise NotImplementedError

  def forward(self, inputs, theta=None):
    ## get theta
    if theta is None:
      theta = self.get_theta()

    ## train
    feature = self.stem(inputs)

    for i, cell in enumerate(self.cells):
      if isinstance(cell, SearchCell):
        feature = cell(feature, theta)
      else:
        feature = cell(feature)

    out = self.lastact(feature)
    out = self.global_pooling( out )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    return logits

  def genotype(self):
    theta = self.get_theta()
    genotypes = []

    for i in range(1, self.max_nodes):
      xlist = []
      for j in range(i):
        node_str = '{:}<-{:}'.format(i, j)
        with torch.no_grad():
          weights = theta[ self.edge2index[node_str] ]
          op_name = self.op_names[ weights.argmax().item() ]
        xlist.append((op_name, j))
      genotypes.append( tuple(xlist) )
    return Structure( genotypes )

  def new(self, enc): # for 2nd order
    # model_new = TinyNetwork(self._C, self._layerN, self.max_nodes, self._num_classes, self._criterion,
                            # self.op_names, self._args, self._affine, self._track_running_stats).cuda()
    model_new = self.get_new_model() # get a new random model
    model_new.set_encoding(enc) # set encoding
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()): # set alpha
      x.data.copy_(y.data)

    return model_new
  
  def is_from_this_supernet(self, theta):
    return theta[self.enc == 0].sum() == 0