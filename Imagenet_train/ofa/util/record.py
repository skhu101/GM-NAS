import time
from queue import deque
from collections import OrderedDict
from contextlib import contextmanager

class RecordManager:

    def __init__(self):
        self.groups = OrderedDict()
        self.records = OrderedDict()
        self.ticks = dict()

    def record_value(self, key, value, window_size, summary_mode, group = 'default'):
        assert window_size == 'inf' or (type(window_size) is int and window_size > 0), \
            'window_size should be positive integer or string "inf"'
        assert summary_mode in ['mean', 'sum', 'max', 'min'], 'summary_mode should be one of mean / sum / max / min'
        Record = {
            'mean': MeanRecord,
            'sum' : SumRecord,
            'max' : MaxRecord,
            'min' : MinRecord,
        }[summary_mode]
        if group not in self.groups:
            self.groups[group] = OrderedDict()
        if key not in self.groups[group]:
            self.groups[group][key] = Record(window_size)
        record = self.groups[group][key]
        assert type(record) is Record, 'renew record %s in group %s with different summary_mode is not allowed' % (repr(key), repr(group))
        assert record.window_size == window_size, 'renew record %s in group %s with different window_size is not allowed' % (repr(key), repr(group))
        record.record_value(value)

    def record_tick(self, key):
        self.ticks[key] = time.time()

    def record_tock(self, key, window_size, summary_mode, group = 'default'):
        if key not in self.ticks:
            raise Exception('key %s hasn\'t been ticked' % repr(key))
        value = time.time() - self.ticks[key]
        self.record_value(key, value, window_size, summary_mode, group)

    @contextmanager
    def record_time(self, key, window_size, summary_mode, group = 'default'):
        self.record_tick(key)
        yield
        self.record_tock(key, window_size, summary_mode, group)


    def summary(self, key, group = 'default'):
        if group not in self.groups:
            raise KeyError('group %s not exists' % repr(group))
        if key not in self.groups[group]:
            raise KeyError('record %s not exists' % repr(key))
        return self.groups[group][key].summary()

    ###########################################################
    # dictionray-like interface
    ###########################################################

    def keys(self, group = 'default'):
        if group not in self.groups:
            raise KeyError('group %s not exists' % repr(group))
        return list(self.groups[group].keys())

    def values(self, group = 'default'):
        if group not in self.groups:
            raise KeyError('group %s not exists' % repr(group))
        return [record.summary() for record in self.groups[group].values()]

    def items(self, group = 'default'):
        if group not in self.groups:
            raise KeyError('group %s not exists' % repr(group))
        return [(key, record.summary()) for key, record in self.groups[group].items()]

    def clear_record(self, key, group = 'default'):
        if group not in self.groups:
            raise KeyError('group %s not exists' % repr(group))
        if key not in self.groups[group]:
            raise KeyError('record %s not exists' % repr(key))
        del self.groups[group][key]

    def clear_records(self, group = 'default'):
        if group not in self.groups:
            raise KeyError('group %s not exists' % repr(group))
        self.groups[group].clear()

class MeanRecord:

    def __init__(self, window_size):
        self.empty = True
        self.window_size = window_size
        if window_size == 'inf':
            self.window = []
        else:
            self.window = deque(maxlen = window_size)

    def record_value(self, value):
        self.empty = False
        self.window.append(value)

    def summary(self):
        assert not self.empty, 'empty record'
        return sum(self.window) / len(self.window)

class SumRecord:

    def __init__(self, window_size):
        self.empty = True
        self.window_size = window_size
        if window_size == 'inf':
            self.value = 0
        else:
            self.window = deque(maxlen = window_size)

    def record_value(self, value):
        self.empty = False
        if self.window_size == 'inf':
            self.value += value
        else:
            self.window.append(value)

    def summary(self):
        assert not self.empty, 'empty record'
        if self.window_size == 'inf':
            return value
        else:
            return sum(self.window)

class _MinMaxRecord:

    def __init__(self, window_size):
        self.window_size = window_size
        self.empty = True
        if window_size != 'inf':
            self.queue = deque(maxlen = window_size)

    def record_value(self, value):
        self.empty = False
        if hasattr(self, 'queue'): self.queue.append(value)
        elif hasattr(self, 'value'): self.value = self.minmax(self.value, value)
        else: self.value = value

    def summary(self):
        assert not self.empty, 'empty record'
        if hasattr(self, 'queue'): return self.minmax(self.queue)
        return self.value

class MinRecord(_MinMaxRecord):
    minmax = min

class MaxRecord(_MinMaxRecord):
    minmax = max
