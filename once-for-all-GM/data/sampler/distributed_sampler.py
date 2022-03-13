__all__ = ['DistributedSampler']

import math
import torch
import distributed as alt_dist
from torch.utils.data.sampler import Sampler

class DistributedSampler(Sampler):

    def __init__(self, dataset, shuffle = False, psudo_index = None):
        self.dataset = dataset
        self.shuffle = shuffle
        self.psudo_index = psudo_index

        self.world_size = alt_dist.get_world_size()
        self.rank = alt_dist.get_rank()
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        self.total_size = self.num_samples * self.world_size

        if self.shuffle:
            self.g = torch.Generator()
            state = torch.get_rng_state()
            # synchronize local random state
            backend = alt_dist.get_backend()
            assert backend == 'nccl', 'invalid backend %s (only support nccl backend currently)' % repr(backend)
            state = state.cuda()
            alt_dist.broadcast([state], 0)
            state = state.cpu()
            self.g.set_state(state)

    def __iter__(self):
        if self.shuffle:
            # shuffle based on (already synchronized) local random generator
            indices = list(torch.randperm(len(self.dataset), generator = self.g))
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        num_extra = self.total_size - len(indices)
        if self.psudo_index is None:
            indices += indices[:num_extra]
        else:
            indices += [self.psudo_index] * num_extra
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
