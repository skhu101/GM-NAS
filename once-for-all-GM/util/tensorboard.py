__all__ = ['init_tensorboard', 'anchor_tensorboard', 'truncate_tensorboard']

import struct
from glob import glob
from pathlib import Path
from .debug import alpha_print
from tensorboardX import SummaryWriter


def init_tensorboard(log_dir, trunc_anchor):
    if trunc_anchor > 0:
        truncate_tensorboard(log_dir, trunc_anchor)
    writer = SummaryWriter(log_dir)
    return writer


def anchor_tensorboard(writer, anchor):
    writer.add_scalar('add-by-alphatrion/anchor', anchor, anchor)


def truncate_tensorboard(log_dir, trunc_anchor):
    log_dir = Path(log_dir)
    for event_fpath in log_dir.glob('events*'):
        f = event_fpath.open('rb+')
        trunc_length = 0
        while True:
            length = f.read(8)
            if len(length) == 0: break
            length = struct.unpack('Q', length)[0]
            f.read(4)
            event = f.read(length)
            # TODO: BY huang
            # event = src.event_pb2.Event.FromString(event)
            f.read(4)
            trunc_length += 16 + length
            try:
                tag = event.summary.value[0].tag
                value = event.summary.value[0].simple_value
                if tag == 'add-by-alphatrion/anchor' and value == trunc_anchor:
                    break
            except:
                pass

        f.seek(0)
        f.truncate(trunc_length)
        f.close()
#        alpha_print('[%s] truncated at length %d / anchor %d' % (
#            event_fpath.relative_to(log_dir.parent.parent),
#            trunc_length, event.step,
#        ), flush=True)
