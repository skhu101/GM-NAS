import pdb
import torch
import torch.nn as nn
from copy import deepcopy
from .cell_operations import ResNetBasicblock
from .search_cells     import NAS201SearchCell as SearchCell
from .genotypes        import Structure
from .search_model_ws import TinyNetworkWS
from torch.autograd import Variable


class TinyNetworkSNASWS(TinyNetworkWS):
  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space, args,
               affine=False, track_running_stats=True):
    super(TinyNetworkSNASWS, self).__init__(C, N, max_nodes, num_classes, criterion, search_space, args,
          affine=affine, track_running_stats=track_running_stats)

    #### log alpha
    self._arch_parameters = nn.Parameter(torch.zeros(self.num_edge, len(search_space)).normal_(1, 0.01).requires_grad_())

    self.tau = args.tau_max
    self.temp_scheduler = None

  def show_alphas(self):
    with torch.no_grad():
      return 'arch-parameters :\n{:}'.format( nn.functional.softmax(self._arch_parameters, dim=-1).cpu() )

  def get_theta(self):
    if self.mode == 'snas':
      return self._get_gumbel_softmax()
    else:
      raise NotImplementedError
    
  def _get_gumbel_softmax(self):
    theta = torch.zeros_like(self._arch_parameters)
    for eid in range(theta.shape[0]):
      theta_eid = self._get_gumbel_softmax_edge(eid)
      theta[eid][self.enc[eid] != 0] = theta_eid
    return theta

  def _get_gumbel_softmax_edge(self, eid):
    log_alpha = self._arch_parameters[eid][self.enc[eid] != 0]
    while True:
      gumbels = torch.nn.functional.gumbel_softmax(log_alpha, tau=self.tau)
      if torch.isinf(gumbels).any():
        continue
      else: break
    return gumbels

  def genotype(self):
    theta = torch.softmax(self._arch_parameters, dim=-1) * self.enc
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