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


def adjust_learning_rate(optimizer, epoch, 
            learning_rate=None, interval=None, epochs=None, decay=.1):
    if interval is not None:
        learning_rate *= (decay ** (epoch // interval))
    else:
        for decay_epoch in epochs:
            if decay_epoch <= epoch:
                learning_rate *= decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
    return

def ensure_file(filename):
    assert os.path.isfile(filename), '{} is not a valid file.'.format(filename)


def ensure_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def cocoval(ann_file, res_file, ann_type='bbox'):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(ann_file)
    coco_dt = coco_gt.loadRes(res_file)
    # imgIds = sorted(coco_gt.getImgIds())
    coco_eval = COCOeval(coco_gt, coco_dt)
    coco_eval.params.useSegm = (ann_type == 'segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
