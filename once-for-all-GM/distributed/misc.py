import os
import math
import torch
import multiprocessing
from util import alpha_print
import torch.distributed as dist


def get_world_size():
    return int(os.environ['SLURM_NTASKS'])


def get_rank():
    return int(os.environ['SLURM_PROCID'])


def get_jobid():
    return int(os.environ['SLURM_JOBID'])


def get_backend():
    return os.environ.get('DISTRIBUTED_BACKEND', None)


# work as a virtual barrier
def barrier():
    if get_world_size() > 1:
        sync_tensor = torch.ones(1).cuda()
        all_reduce(sync_tensor)
        sync_value = sync_tensor.item()


def all_reduce_mean(tensor_list):
    if get_world_size() == 1: return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.reduce_op.SUM)
        tensor.div_(get_world_size())


def all_reduce_sum(tensor_list):
    if get_world_size() == 1: return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.reduce_op.SUM)


def all_reduce_max(tensor_list):
    if get_world_size() == 1: return
    for tensor in tensor_list:
        dist.all_reduce(tensor, op=dist.reduce_op.MAX)


def all_reduce_min(tensor_list):
    if get_world_size() == 1: return
    for tensor in tensor_list:
        tensor.neg_()
        dist.all_reduce(tensor, op=dist.reduce_op.MAX)
        tensor.neg_()


def broadcast(tensor_list, src):
    if get_world_size() == 1: return
    for tensor in tensor_list:
        dist.broadcast(tensor, src)


def dist_segment(full_size, world_size=None, rank=None):
    if world_size is None:
        world_size = get_world_size()
    if rank is None:
        rank = get_rank()
    interval = math.ceil(full_size / world_size)
    offset = interval * rank
    part_size = min(full_size, offset + interval) - offset
    return offset, part_size


def dist_init(port, backend):
    os.environ['DISTRIBUTED_BACKEND'] = backend

    rank = get_rank()
    world_size = get_world_size()

    addr = None

    num_gpus = torch.cuda.device_count()
    print("num_gpus", num_gpus)
    gpu_id = rank % num_gpus

    torch.cuda.set_device(gpu_id)

    if world_size == 1:
        rank, world_size = 0, 1
        alpha_print('using single card, no distributed environment init', flush=True)
    else:
        os.environ['MASTER_PORT'] = str(port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        dist.init_process_group(backend=backend)

        form = '%%%dd' % len(str(world_size))
        alpha_print('world_size %d, distributed init rank %s, gpu %d, at %s:%d' % (
            world_size, form % rank, gpu_id, addr, port
        ), flush=True)

    return rank, world_size

def dist_init_pytorch(port, backend, rank):
    os.environ['DISTRIBUTED_BACKEND'] = backend
    addr = None
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['SLURM_NTASKS']= str(os.environ['WORLD_SIZE'])
    os.environ['SLURM_PROCID']= str(rank)
    num_gpus = torch.cuda.device_count()
    print("num_gpus", num_gpus)
    gpu_id = rank % num_gpus

    torch.cuda.set_device(gpu_id)

    if world_size == 1:
        rank, world_size = 0, 1
        alpha_print('using single card, no distributed environment init', flush=True)
    else:
        os.environ['MASTER_PORT'] = str(port)
        
        os.environ['RANK'] = str(rank)
        dist.init_process_group(backend=backend)

        form = '%%%dd' % len(str(world_size))
        alpha_print('world_size %d, distributed init rank %s, gpu %d, at %s:%d' % (
            world_size, form % rank, gpu_id, addr, port
        ), flush=True)

    return rank, world_size



