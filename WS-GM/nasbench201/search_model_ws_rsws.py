import pdb
import torch
import torch.nn as nn
from .search_model import TinyNetwork as TinyNetwork
from .search_model_ws import TinyNetworkWS
from .genotypes        import Structure



class TinyNetworkRSWS(TinyNetworkWS):
  def __init__(self, C, N, max_nodes, num_classes, criterion, search_space, args,
               affine=False, track_running_stats=True):
    super(TinyNetworkRSWS, self).__init__(C, N, max_nodes, num_classes, criterion, search_space, args,
          affine=affine, track_running_stats=track_running_stats)

  def genotype(self): # upon calling, the caller should pass the "theta" into this object as "alpha" first
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