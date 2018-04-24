from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable, grad
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
    parser.add_argument('--ann_file', default='',
                        help='where you put official json')
    parser.add_argument('--dt_file', default='',
                        help='where you put our json')
    parser.add_argument('--resume', default="",
                        help='where you save checkpoint')
    parser.add_argument('--pretrain', default='',
                        help='where you save pretrain model')
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=5, type=int)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--evaluate', dest='evaluate',
                        help='evaluate mode',
                        action='store_true')

    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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
    dataset = COCODataset(
            args.data_dir, 
            args.ann_file, 
            args.dt_file, 
            COCOTransform([800], 1200, flip=False),
            normalize_fn=normalize)
    dataloader = COCODataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=6)

    model = resnet101()
    logger.info(model)

    if args.pretrain:
        assert os.path.isfile(args.pretrain), '{} is not a valid file'.format(args.pretrain)
        checkpoint = torch.load(args.pretrain)
        model.load_state_dict(checkpoint, strict=False)

    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    model.freeze_layer()
    logger.info("Layers already freezed.")

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
            adjust_learning_rate(optimizer, epoch, args.lr, interval=4)
            Train(epoch, model, dataloader, optimizer, logger)
            Savecheckpoint(epoch, model)
    logger.info('Exit without error.')


def Savecheckpoint(epoch, model):
    if args.mGPUs:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save({
            'epoch': epoch,
            'state_dict': model_state_dict}, 
        os.path.join(args.save_dir, 'epoch_%d.pth' % (epoch + 1)) )


def Evaluate(model, val_loader, logger):
    losses = AveMeter(100)
    batch_time = AveMeter(100)
    data_time = AveMeter(100)
    Prec1 = AveMeter(len(val_loader))
    Prec5 = AveMeter(len(val_loader))
    model.eval()

    start = time.time()
    for i, inp in enumerate(val_loader):
        # input data processing
        img_var = inp[0].cuda(async=True)
        bboxes = Variable(inp[1][:,:,:5].contiguous()).cuda(async=True)
        targets = Variable(inp[2][:,:,:,1].contiguous()).cuda(async=True)
        weights = Variable(inp[2][:,:,:,2].contiguous()).cuda(async=True)
        data_time.add(time.time() - start)

        # forward
        pred, loss, _ = model(img, bboxes, targets, weights)
        loss = loss.mean()

        # get output datas
        pred = pred.cpu().data.numpy()
        targets = targets.cpu().data.numpy()

        losses.add(loss.data[0])
        Prec1.add(accuracy(pred, targets, 1))
        Prec5.add(accuracy(pred, targets, 5))
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
    logger.info('Prec1: %.3f Prec5: %.3f' % (Prec1.avg, Prec5.avg))
    

def Train(epoch, model, train_loader, optimizer, logger):
    losses = AveMeter(100)
    noweight_losses = AveMeter(100)
    batch_time = AveMeter(100)
    data_time = AveMeter(100)
    model.train()

    start = time.time()
    for i, inp in enumerate(train_loader):
        # input data processing
        img_var = inp[0].cuda(async=True)
        bboxes = Variable(inp[1][:,:,:5].contiguous()).cuda(async=True)
        targets = Variable(inp[2][:,:,:,1].contiguous()).cuda(async=True)
        weights = Variable(inp[2][:,:,:,2].contiguous()).cuda(async=True)
        data_time.add(time.time() - start)

        # forward
        pred, loss, noweight_loss = model(img_var, bboxes, targets, weights)
        loss, noweight_loss = loss.mean(), noweight_loss.mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.add(loss.data[0])
        noweight_losses.add(noweight_loss.data[0])
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
