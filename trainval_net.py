from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
torch.backends.cudnn.enabled = False
from torch.autograd import Variable, grad
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import logging

from datasets.RL_coco_dataset import COCODataset, COCOTransform
from datasets.RL_coco_loader import COCODataLoader
from model.Reinforcement.resnet import resnet101
from model.Reinforcement.utils import init_log, AveMeter, accuracy, adjust_learning_rate

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--data_dir', default='',
                        help='where you put images')
    parser.add_argument('--anno_file', default='',
                        help='where you put official json')
    parser.add_argument('--labels_file', default='',
                        help='where you put our json')
    parser.add_argument('--resume', default="",
                        help='where you save checkpoint')
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=8, type=int)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--evaluate', dest='evaluate',
                        help='evaluate mode',
                        action='store_true')

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--batch-size', default=24, type=int,
                        help="batch_size (default: 24)")

    args = parser.parse_args()
    return args

def main():
    init_log('global', logging.INFO)
    logger = logging.getLogger('global')

    args = parse_args()
    logger.info(args)

    normalize = transforms.Normalize(mean=[0.4485295, 0.4249905, 0.39198247],
                                     std=[0.12032582, 0.12394787, 0.14252729])
    dataset = COCODataset(args.data_dir, args.anno_file, args.labels_file, COCOTransform([800], 1200, flip=False),
                          normalize_fn=normalize)
    logger.info("Dataset Build Done")

    dataloader = COCODataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    logger.info("DataLoader Build Done")

    model = resnet101()
    logger.info("Model Build Done")
    logger.info(model)

    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint, strict=False)

    model.freeze_layer()
    logger.info("Freeze Done")

    params = []
    for i, (key, value) in enumerate(dict(model.named_parameters()).items()):
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': args.lr * .5, 'weight_decay': 0}]
            else:
                params += [{'params': [value], 'lr': args.lr, 'weight_decay': args.weight_decay}]

    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.mGPUs:
        model = nn.DataParallel(model)

    model = model.cuda()

    if args.evaluate:
        Evaluate(model, dataloader, logger)
    else:
        for epoch in range(args.max_epochs):
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict()}, 
                'snapshot/epoch_%d.pth' % epoch)
            adjust_learning_rate(optimizer, epoch, args.lr, interval=5)
            Train(epoch, model, dataloader, optimizer, logger)
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict()}, 
                'snapshot/epoch_%d.pth' % (epoch + 1))

def Evaluate(model, val_loader, logger):
    losses = AveMeter(100)
    batch_time = AveMeter(100)
    data_time = AveMeter(100)
    Prec1 = AveMeter(len(val_loader))
    Prec5 = AveMeter(len(val_loader))
    model.eval()

    start = time.time()
    for i, inp in enumerate(val_loader):
        x = {
            'image': torch.autograd.Variable(inp[0]).cuda(),
            'bbox': torch.autograd.Variable(torch.FloatTensor(inp[2])).cuda(),
        }
        data_time.add(time.time() - start)

        pred, loss, _, label= model(x)
        losses.add(loss.data[0])

        pred = pred.cpu().data.numpy()
        label = label.cpu().data.numpy()

        Prec1.add(accuracy(pred, label, 1))
        Prec5.add(accuracy(pred, label, 5))

        batch_time.add(time.time() - start)
        logger.info('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Prec1 {Prec1.val:.3f} ({Prec1.avg:.3f})\t'
                    'Prec5 {Prec5.val:.3f} ({Prec5.avg:.3f})\t'.format(
                        i, len(val_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        Prec1=Prec1,
                        Prec5=Prec5)
                    )
        start = time.time()
    logger.info('Prec1: %.3f Prec3: %.3f' % (Prec1.avg, Prec3.avg))
    

def Train(epoch, model, train_loader, optimizer, logger):
    losses = AveMeter(100)
    noweight_losses = AveMeter(100)
    batch_time = AveMeter(100)
    data_time = AveMeter(100)
    model.train()

    start = time.time()
    for i, inp in enumerate(train_loader):
        x = {
            'image': torch.autograd.Variable(inp[0]).cuda(),
            'bbox': torch.autograd.Variable(torch.FloatTensor(inp[2])).cuda(),
        }
        data_time.add(time.time() - start)

        pred, loss, noweight_loss, label = model(x)

        losses.add(loss.data[0])
        noweight_losses.add(noweight_loss.data[0])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.add(time.time() - start)
        logger.info('Train: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                    'NWLoss {nwlosses.val:.3f} ({nwlosses.avg:.3f})\t'.format(
                        epoch + 1, i, len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        losses=losses,
                        nwlosses=noweight_losses)
                    )
        start = time.time()

if __name__ == "__main__":
    main()
