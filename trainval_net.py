from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import time
import logging
import argparse
import numpy as np
from config import Config

import torch
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler

from datasets.RL_coco_dataset import COCODataset, COCOTransform
from datasets.RL_coco_loader import COCODataLoader
from model.Reinforcement.resnet import resnet101
from model.Reinforcement.action import Action
from model.Reinforcement.utils import *

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
	parser.add_argument('-sd', '--save-dir', type=str,
						default='', help='where you save checkpoint')
	parser.add_argument('-re', '--resume', type=str,
						default='', help='enable the continue from a specific model')

	parser.add_argument('--test', dest='test', action='store_true',
						help='enable the test mode')
	parser.add_argument('--mGPUs', dest='mGPUs', action='store_true',
						help='whether use multiple GPUs')

	parser.add_argument('-e', '--epoch', default=0, type=int,
						help='test model epoch num (default: 0)')
	parser.add_argument('-b', '--batch-size', default=24, type=int,
						help='batch_size (default: 24)')
	parser.add_argument('--log-interval', default=10, type=int,
						help='iter logger info interval (default: 10)')

	args = parser.parse_args()
	return args

def main():
	global args, config

	# initialize logger 
	init_log('global', logging.INFO)
	logger = logging.getLogger('global')
	
	# initialize arguments
	args = parse_args()
	logger.info(args)
	phase = 'minival' if args.test else 'train'
	logger.info('Now using phase: ' + phase)

	# initialize config
	config = Config(phase=phase)

	# create actions
	bbox_action = Action(delta=config.act_delta,
						iou_thres=config.act_iou_thres,
						wtrans=config.act_wtrans)
	# create data_loader
	normalize_fn = config.normalize
	if phase == 'train':
		transform_fn = COCOTransform(config.train_img_short, config.train_img_size, flip=config.train_flip) 
	else:
		transform_fn = COCOTransform(config.test_img_short, config.test_img_size, flip=config.test_flip)
	dataset = COCODataset(
		config.data_dir, 
		config.ann_file, 
		config.dt_file,
		bbox_action=bbox_action,
		transform_fn=transform_fn,
		normalize_fn=normalize_fn)
	dataloader = COCODataLoader(
		dataset, 
		batch_size=args.batch_size, 
		shuffle=config.data_shuffle, 
		num_workers=config.num_workers,
		pin_memory=config.data_pin_memory)

	# create model
	model = resnet101(num_acts=bbox_action.num_acts)
	logger.info(model)

	# load pretrained model
	if config.pretrained_model:
		ensure_file(config.pretrained_model)
		checkpoint = torch.load(config.pretrained_model)
		model.load_state_dict(checkpoint, strict=False)

	# adjust learning rate of layers in the model
	model.freeze_layer()
	params = []
	for i, (key, value) in enumerate(dict(model.named_parameters()).items()):
		if value.requires_grad:
			if 'bias' in key:
				params += [{'params': [value], 'lr': config.learning_rate * 2., 'weight_decay': 0}]
			else:
				params += [{'params': [value], 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]

	# SGD optimizer
	optimizer = torch.optim.SGD(params, config.learning_rate,
								momentum=config.momentum,
								weight_decay=config.weight_decay)

	# enable multi-GPU training
	if args.mGPUs:
		model = nn.DataParallel(model)

	model = model.cuda()

	# ensure save directory
	save_dir = args.save_dir if args.save_dir else config.save_dir
	ensure_dir(save_dir)
	# main function
	if phase == 'train':
		start_epoch = 0
		# if continue from a breakpoint
		if args.resume:
			ensure_file(args.resume)
			checkpoint = torch.load(args.resume)
			start_epoch = checkpoint['epoch']
			model.load_state_dict(checkpoint['state_dict'], strict=False)
		# start training
		for epoch in range(start_epoch, config.train_max_epoch):
			adjust_learning_rate(optimizer, epoch, 
				learning_rate=config.learning_rate, 
				epochs=config.train_lr_decay)
			#
			Train(epoch, model, dataloader, optimizer)
			Savecheckpoint(save_dir, epoch, model)
	else:
		resume = save_dir + 'epoch_{}.pth'.format(args.epoch)
		ensure_file(resume)
		checkpoint = torch.load(resume)
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'], strict=False)
		#
		dt_boxes = Evaluate(model, dataloader, bbox_action)
		ensure_dir(os.path.join(args.save_dir, 'jsons'))
		filename = 'detections_{}_epoch{}_results.json'.format((args.phase, start_epoch))
		filename = os.path.join(args.save_dir, 'jsons', filename)
		json.dump(dt_boxes, open(filename, 'w'))
		cocoval(args.ann_file, filename)
		
	logger.info('Exit without error.')


def Savecheckpoint(save_dir, epoch, model):
	global args, config

	# model differ when on multi-GPU
	if args.mGPUs:
		model_state_dict = model.module.state_dict()
	else:
		model_state_dict = model.state_dict()

	# save checkpoint
	torch.save({
			'epoch': epoch,
			'state_dict': model_state_dict}, 
		os.path.join(save_dir, 'epoch_%d.pth' % (epoch + 1)) )


def Evaluate(model, val_loader, bbox_action):
	global args, config
	logger = logging.getLogger('global')

	losses = AveMeter(100)
	batch_time = AveMeter(100)
	data_time = AveMeter(100)
	#Prec1 = AveMeter(len(val_loader))
	#Prec5 = AveMeter(len(val_loader))
	Preck = AveMeter(len(val_loader))
	dt_boxes = []
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
		pred, loss, _ = model(img_var, bboxes, targets, weights)
		loss = loss.mean()

		# get output boxes
		bboxes = inp[1].numpy()
		bboxes[:, :, 3] = bboxes[:, :, 3] - bboxes[:, :, 1]
		bboxes[:, :, 4] = bboxes[:, :, 4] - bboxes[:, :, 2]
		batch_size = bboxes.shape[0]

		# get output datas
		preds = pred.cpu().data.numpy().reshape(batch_size, -1, bbox_action.num_acts)
		targets = targets.cpu().data.numpy().reshape(batch_size, -1, bbox_action.num_acts)

		# get new boxes
		newboxes, preck = bbox_action.move_from_act(bboxes[:,:,1:5], preds, targets, maxk=1)
		bboxes[:,:,1:5] = newboxes
		bboxes = bboxes.reshape(-1, bboxes.shape[-1]).astype(float)

		# generate detection results
		im_infos = inp[3]
		for j, bbox in enumerate(bboxes):
			bid = int(bbox[0])
			scale = im_infos[bid][2]
			bbox[1:5] /= scale
			
			dtbox = {
				'bbox': [float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])],
				'score': float(bbox[5]),
				'category_id': int(bbox[6]),
				'image_id': int(bbox[7])
			}
			'''
			dtbox = {
				'bbox': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
				'score': float(bbox[4]),
				'category_id': int(bbox[5]),
				'image_id': int(bbox[6])
			}
			'''
			dt_boxes.append(dtbox)

		losses.add(loss.item())
		#Prec1.add(accuracy(preds, targets, 1))
		Preck.add(preck)
		batch_time.add(time.time() - start)
		if i % args.log_interval == 0:
			logger.info('Test: [{0}/{1}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
						'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
						#'Prec1 {Prec1.val:.3f} ({Prec1.avg:.3f})\t'
						'Preck {Preck.val:.3f} ({Preck.avg:.3f})\t'
						#'Prec5 {Prec5.val:.3f} ({Prec5.avg:.3f})\t'
						.format(i, len(val_loader),
							batch_time=batch_time,
							data_time=data_time,
							losses=losses,
							#Prec1=Prec1,
							#Prec5=Prec5,
							Preck=Preck)
						)
		start = time.time()
	#logger.info('Prec1: %.3f Prec5: %.3f' % (Prec1.avg, Prec5.avg))
	logger.info('Preck: %.3f' % (Preck.avg))
	return dtboxes


def Train(epoch, model, train_loader, optimizer):
	global args, config
	logger = logging.getLogger('global')

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
		
		losses.add(loss.item())
		noweight_losses.add(noweight_loss.item())
		batch_time.add(time.time() - start)
		if i % args.log_interval == 0:
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
