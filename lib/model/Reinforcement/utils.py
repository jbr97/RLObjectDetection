import logging
import os
import numpy as np

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

def transform(bboxes, actions):
    """
    :param bboxes:
    :param actions:
    :return:
    """
    #assert bboxes.shape[0] == len(actions), 'Unmatched bboxes and actiosn.'

    #transform_bboxes = bboxes.copy()
    #for i, action in enumerate(actions):
    #    action = action - 1

    #    x, y, x2, y2= transform_bboxes[i, 1:5]
    #    w = x2 - x
    #    h = y2 - y

    #    # 1-7: [x,y,w,h] -> [x+0.5w, y, w, h]
    #    if action == 0:   x += w * 0.5**1
    #    elif action == 1: x += w * 0.5**2
    #    elif action == 2: x += w * 0.5**3
    #    elif action == 3: x += w * 0.5**4
    #    elif action == 4: x += w * 0.5**5
    #    elif action == 5: x += w * 0.5**6
    #    elif action == 6: x += w * 0.5**7
    #    # 8-14: [x,y,w,h] -> [x, y+0.5h, w, h]
    #    elif action == 7:  y += h * 0.5**1
    #    elif action == 8:  y += h * 0.5**2
    #    elif action == 9: y += h * 0.5**3
    #    elif action == 10: y += h * 0.5**4
    #    elif action == 11: y += h * 0.5**5
    #    elif action == 12: y += h * 0.5**6
    #    elif action == 13: y += h * 0.5**7
    #    # 15-21: [x,y,w,h] -> [x, y, w+0.5w, h]
    #    elif action == 14: w += w * 0.5**1
    #    elif action == 15: w += w * 0.5**2
    #    elif action == 16: w += w * 0.5**3
    #    elif action == 17: w += w * 0.5**4
    #    elif action == 18: w += w * 0.5**5
    #    elif action == 19: w += w * 0.5**6
    #    elif action == 20: w += w * 0.5**7
    #    # 22-28: [x,y,w,h] -> [x, y, w, h+0.5h]
    #    elif action == 21: h += h * 0.5**1
    #    elif action == 22: h += h * 0.5**2
    #    elif action == 23: h += h * 0.5**3
    #    elif action == 24: h += h * 0.5**4
    #    elif action == 25: h += h * 0.5**5
    #    elif action == 26: h += h * 0.5**6
    #    elif action == 27: h += h * 0.5**7
    #    # 29-35: [x,y,w,h] -> [x-0.5w, y, w, h]
    #    elif action == 28: x -= w * 0.5**1
    #    elif action == 29: x -= w * 0.5**2
    #    elif action == 30: x -= w * 0.5**3
    #    elif action == 31: x -= w * 0.5**4
    #    elif action == 32: x -= w * 0.5**5
    #    elif action == 33: x -= w * 0.5**6
    #    elif action == 34: x -= w * 0.5**7
    #    # 36-42: [x,y,w,h] -> [x, y-0.5h, w, h]
    #    elif action == 35: y -= h * 0.5**1
    #    elif action == 36: y -= h * 0.5**2
    #    elif action == 37: y -= h * 0.5**3
    #    elif action == 38: y -= h * 0.5**4
    #    elif action == 39: y -= h * 0.5**5
    #    elif action == 40: y -= h * 0.5**6
    #    elif action == 41: y -= h * 0.5**7
    #    # 43-49: [x,y,w,h] -> [x, y, w-0.5w, h]
    #    elif action == 42: w -= w * 0.5**1
    #    elif action == 43: w -= w * 0.5**2
    #    elif action == 44: w -= w * 0.5**3
    #    elif action == 45: w -= w * 0.5**4
    #    elif action == 46: w -= w * 0.5**5
    #    elif action == 47: w -= w * 0.5**6
    #    elif action == 48: w -= w * 0.5**7
    #    # 22,23,24: [x,y,w,h] -> [x, y, w, h-0.5h]
    #    elif action == 49: h -= h * 0.5**1
    #    elif action == 50: h -= h * 0.5**2
    #    elif action == 51: h -= h * 0.5**3
    #    elif action == 52: h -= h * 0.5**4
    #    elif action == 53: h -= h * 0.5**5
    #    elif action == 54: h -= h * 0.5**6
    #    elif action == 55: h -= h * 0.5**7

    #    action += 1
    #    transform_bboxes[i, 1:5] = np.array([x, y, x+w, y+h])
    transform_bboxes = bboxes.copy()
    for i, action in enumerate(actions):
        if action == 1:
            transform_bboxes[i, 1] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.02
        elif action == 2:
            transform_bboxes[i, 1] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.05
        elif action == 3:
            transform_bboxes[i, 1] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.1
        elif action == 4:
            transform_bboxes[i, 2] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.02
        elif action == 5:
            transform_bboxes[i, 2] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.05
        elif action == 6:
            transform_bboxes[i, 2] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.1
        elif action == 7:
            transform_bboxes[i, 3] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.02
        elif action == 8:
            transform_bboxes[i, 3] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.05
        elif action == 9:
            transform_bboxes[i, 3] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.1
        elif action == 10:
            transform_bboxes[i, 4] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.02
        elif action == 11:
            transform_bboxes[i, 4] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.05
        elif action == 12:
            transform_bboxes[i, 4] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.1
        elif action == 13:
            transform_bboxes[i, 1] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.02
        elif action == 14:
            transform_bboxes[i, 1] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.05
        elif action == 15:
            transform_bboxes[i, 1] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.1
        elif action == 16:
            transform_bboxes[i, 2] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.02
        elif action == 17:
            transform_bboxes[i, 2] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.05
        elif action == 18:
            transform_bboxes[i, 2] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.1
        elif action == 19:
            transform_bboxes[i, 3] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.02
        elif action == 20:
            transform_bboxes[i, 3] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.05
        elif action == 21:
            transform_bboxes[i, 3] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.1
        elif action == 22:
            transform_bboxes[i, 4] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.02
        elif action == 23:
            transform_bboxes[i, 4] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.05
        elif action == 24:
            transform_bboxes[i, 4] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.1
    return transform_bboxes
