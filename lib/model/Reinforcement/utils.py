import logging
import os
import numpy as np
from collections import deque

logs = set()

def init_log(name, level = logging.INFO):
    if (name, level) in logs: return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = '%(asctime)s-rk{}-%(filename)s#%(lineno)d:%(message)s'.format(rank)
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class AveMeter():
    def __init__(self, size):
        self.size = size
        self.opr = 0
        self.val = 0.0
        self.avg = 0.0
        self.elems = list()

    def add(self, x):
        self.val = x
        if self.opr >= self.size:
            pos = self.opr % self.size
            self.elems[pos] = x
            self.avg = sum(self.elems) / self.size
        else:
            self.elems.append(x)
            self.avg = sum(self.elems) / (self.opr+1)
        self.opr += 1


class Counter(object):
    def __init__(self, size=1000):
        self._size = size
        self.items = deque(list(), self._size)

    def add(self, x) :
        if isinstance(x, list):
            self.items.extend(x)
        else:
            self.items.append(x)

    def __len__(self):
        return len(self.items)

    def get_statinfo(self):
        # assert len(self.items) == self._size, 'not enough sample in the counter.'
        
        sorted_items = sorted(self.items)
        nums = len(self.items)
        a, b, c, d, e = 0, int(nums/4), int(nums/2), int(nums/4*3), nums-1  

        return sorted_items[a], sorted_items[b], sorted_items[c], sorted_items[d], sorted_items[e]


def accuracy(output, target, k=1):
    output, target = output.reshape(-1), target.reshape(-1)
    inds = np.argsort(output)[-k:]

    output = output[inds]
    target = target[inds]
    correct = np.sum(target == 1)

    return correct * 100.0 / k


def adjust_learning_rate(optimizer, epoch, lr, interval=None, epochs=None, alpha=.1):
    if interval is not None:
        lr = lr * (alpha ** (epoch // interval))
    else:
        for i in epochs:
            if i >= epoch:
                lr *= alpha
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return
