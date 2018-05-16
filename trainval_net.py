from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import time
import json
import math
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
	config = Config()

	# create actions
	bbox_action = Action(delta=config.act_delta,
						iou_thres=config.act_iou_thres,
						wtrans=config.act_wtrans)

	# create train_loader
	normalize_fn = config.normalize
	if phase == 'train':
		train_transform_fn = COCOTransform(config.train_img_short, config.train_img_size, flip=config.train_flip) 
		train_dataset = COCODataset(
			config.train_data_dir, 
			config.train_ann_file, 
			config.train_dt_file,
			bbox_action=bbox_action,
			transform_fn=train_transform_fn,
			normalize_fn=normalize_fn,
			static_file='log/train_static.npy',
			phase=phase)
		train_loader = COCODataLoader(
			train_dataset, 
			batch_size=args.batch_size, 
			shuffle=config.data_shuffle, 
			num_workers=config.num_workers,
			pin_memory=config.data_pin_memory)
	else:
		# create test_loader
		test_transform_fn = COCOTransform(config.test_img_short, config.test_img_size, flip=config.test_flip)
		test_dataset = COCODataset(
			config.test_data_dir, 
			config.test_ann_file, 
			config.test_dt_file,
			bbox_action=bbox_action,
			transform_fn=test_transform_fn,
			normalize_fn=normalize_fn,
			static_file='log/test_static.npy',
			phase=phase)
		test_loader = COCODataLoader(
			test_dataset, 
			batch_size=args.batch_size, 
			shuffle=config.data_shuffle, 
			num_workers=config.num_workers,
			pin_memory=config.data_pin_memory)
		cls2cat = test_dataset.cls2cat

	# create model
	model = resnet101(
		num_acts=bbox_action.num_acts, 
		num_classes=config.num_classes)
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
			Train(epoch, model, train_loader, optimizer)
			Savecheckpoint(save_dir, epoch, model)
	else:
		resume = save_dir + 'epoch_{}.pth'.format(args.epoch)
		ensure_file(resume)
		checkpoint = torch.load(resume)
		start_epoch = checkpoint['epoch']
		model.load_state_dict(checkpoint['state_dict'], strict=False)
		#
		dt_boxes = Evaluate(model, test_loader, bbox_action, cls2cat)
		ensure_dir(os.path.join(save_dir, 'jsons'))
		filename = 'detections_{}_epoch{}_results.json'.format(phase, start_epoch)
		filename = os.path.join(save_dir, 'jsons', filename)
		json.dump(dt_boxes, open(filename, 'w'))
		cocoval(config.test_ann_file, filename)
		
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


def Evaluate(model, val_loader, bbox_action, cls2cat):
	global args, config
	logger = logging.getLogger('global')

	losses = AveMeter(100)
	batch_time = AveMeter(100)
	data_time = AveMeter(100)
	Prec_per_box = AveMeter(len(val_loader))
	Prec1_per_img = AveMeter(len(val_loader))
	Prec5_per_img = AveMeter(len(val_loader))
	Prec10_per_img = AveMeter(len(val_loader))
	Prec1_per_act = AveMeter(bbox_action.num_acts)
	Prec5_per_act = AveMeter(bbox_action.num_acts)
	Prec10_per_act = AveMeter(bbox_action.num_acts)
	delta_ious = AveMeter(100)
	model.eval()

	start = time.time()
	for i, inp in enumerate(val_loader):
		# input data processing
		img_var = inp[0].cuda(async=True)
		inp1 = inp[1]
		inp2 = inp[2]
		batch_size, num_boxes, _ = inp1.numpy().shape
		'''
		bboxes = Variable(inp[1][:,:,:5].contiguous()).cuda(async=True)
		cls_ids = Variable(inp[1][:,:,6].contiguous()).cuda(async=True)
		new_bboxes = inp2[:,s:e,:,:5].contiguous().cuda(async=True)
		targets = Variable(inp[2][:,:,:,-2].contiguous()).cuda(async=True)
		weights = Variable(inp[2][:,:,:,-1].contiguous()).cuda(async=True)
		'''
		data_time.add(time.time() - start)

		# forward
		#pred, loss, _ = model(img_var, bboxes, cls_ids, targets, weights)
		#loss = loss.mean()

		preds = np.zeros((batch_size, num_boxes, bbox_action.num_acts))
		pnum = 100 // bbox_action.num_acts
		iters = math.ceil(num_boxes / pnum)
		#logger.info(iters)
		input_feature = None
		for this_iter in range(iters):
			s = this_iter * pnum
			e = min((this_iter+1) * pnum, num_boxes)

			bboxes = inp1[:,s:e,:5].contiguous().cuda(async=True)
			cls_ids = inp1[:,s:e,6].contiguous().cuda(async=True)
			new_bboxes = inp2[:,s:e,:,:5].contiguous().cuda(async=True)
			targets = inp2[:,s:e,:,-2].contiguous().cuda(async=True)
			weights = inp2[:,s:e,:,-1].contiguous().cuda(async=True)
			
			#forward
			if input_feature is None:
				pred, loss, input_feature = model(img_var, bboxes, new_bboxes, cls_ids, targets, weights)
			else:
				pred, loss = model(img_var, bboxes, new_bboxes, cls_ids, targets, weights, input_feature)
			loss = loss.mean()
			for act_id in range(bbox_action.num_acts):
				preds[:,s:e,act_id] = pred[act_id].cpu().data.numpy()

			losses.add(loss.item())
			logger.info('this iter:{}, loss: {}'.format(this_iter, loss.item()))

		# get output boxes
		bboxes = inp1.numpy()
		bboxes[:, :, 3] = bboxes[:, :, 3] - bboxes[:, :, 1]
		bboxes[:, :, 4] = bboxes[:, :, 4] - bboxes[:, :, 2]

		# get output datas
		#for act_id in range(bbox_action.num_acts):
			#preds[:,:,act_id] = pred[act_id].cpu().data.numpy().reshape(batch_size, num_boxes)
		#preds = pred.cpu().data.numpy().reshape(batch_size, -1, bbox_action.num_acts)
		targets = inp2[:,:,:,-2].contiguous().numpy().reshape(batch_size, -1, bbox_action.num_acts)
		weights = inp2[:,:,:,-1].contiguous().numpy().reshape(batch_size, -1, bbox_action.num_acts)

		# get precs
		prec_per_box = bbox_action.accuracy_per_box(preds, targets)
		prec1_per_img = bbox_action.accuracy_per_img(preds, targets, maxk=1)
		prec5_per_img = bbox_action.accuracy_per_img(preds, targets, maxk=5)
		prec10_per_img = bbox_action.accuracy_per_img(preds, targets, maxk=10)
		Prec_per_box.add(prec_per_box)
		Prec1_per_img.add(prec1_per_img.mean())
		Prec5_per_img.add(prec5_per_img.mean())
		Prec10_per_img.add(prec10_per_img.mean())

		# collect preds & targets
		if i == 0:
			dt_boxes = []
			all_preds = preds.reshape(-1, bbox_action.num_acts)
			all_targets = targets.reshape(-1, bbox_action.num_acts)
		else:
			all_preds = np.concatenate([all_preds, preds.reshape(-1, bbox_action.num_acts)], axis=0)
			all_targets = np.concatenate([all_targets, targets.reshape(-1, bbox_action.num_acts)], axis=0)

		# get new boxes
		newboxes = bbox_action.move_from_act(bboxes[:,:,1:5], preds, targets, maxk=10)
		#newboxes, delta_iou = bbox_action.move_from_weight(bboxes[:,:,1:5], weights, targets, maxk=10)
		#delta_ious.add(delta_iou)
		#logger.info(delta_ious.avg)
		bboxes[:,:,1:5] = newboxes

		# generate detection results
		bboxes = bboxes.reshape(-1, bboxes.shape[-1]).astype(float)
		im_infos = inp[3]
		for j, bbox in enumerate(bboxes):
			bid = int(bbox[0])
			scale = im_infos[bid][2]
			bbox[1:5] /= scale
			
			dtbox = {
				'bbox': [float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4])],
				'score': float(bbox[5]),
				'category_id': int(cls2cat[int(bbox[6])]),
				'image_id': int(bbox[7])
			}
			dt_boxes.append(dtbox)

		losses.add(loss.item())
		batch_time.add(time.time() - start)
		if i % args.log_interval == 0 or i == len(val_loader)-1:
			logger.info('Test: [{0}/{1}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
						'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
						'Prec_per_box {Prec_per_box.val:.3f} ({Prec_per_box.avg:.3f})\t'
						'Prec1_per_img {Prec1_per_img.val:.3f} ({Prec1_per_img.avg:.3f})\t'
						'Prec5_per_img {Prec5_per_img.val:.3f} ({Prec5_per_img.avg:.3f})\t'
						'Prec10_per_img {Prec10_per_img.val:.3f} ({Prec10_per_img.avg:.3f})\t'.format(
							i, len(val_loader),
							batch_time=batch_time,
							data_time=data_time,
							losses=losses,
							Prec_per_box=Prec_per_box, 
							Prec1_per_img=Prec1_per_img, 
							Prec5_per_img=Prec5_per_img, 
							Prec10_per_img=Prec10_per_img)
						)
		start = time.time()

	prec1_per_act = bbox_action.accuracy_per_act(all_preds, all_targets, ratio=.01)
	prec5_per_act = bbox_action.accuracy_per_act(all_preds, all_targets, ratio=.05)
	prec10_per_act = bbox_action.accuracy_per_act(all_preds, all_targets, ratio=.1)

	for act_id in range(bbox_action.num_acts):
		Prec1_per_act.add(prec1_per_act[act_id])
		Prec5_per_act.add(prec5_per_act[act_id])
		Prec10_per_act.add(prec10_per_act[act_id])
		logger.info('Action id: [{0}]\t'
					'Action Delta: {1}\t'
					'Prec1_per_act {Prec1_per_act.val:.3f} ({Prec1_per_act.avg:.3f})\t'
					'Prec5_per_act {Prec5_per_act.val:.3f} ({Prec5_per_act.avg:.3f})\t'
					'Prec10_per_act {Prec10_per_act.val:.3f} ({Prec10_per_act.avg:.3f})\t'.format(
						act_id, str(bbox_action.actDeltas[act_id]),
						Prec1_per_act=Prec1_per_act,
						Prec5_per_act=Prec5_per_act,
						Prec10_per_act=Prec10_per_act)
					)
	return dt_boxes


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
		inp1 = inp[1]
		inp2 = inp[2]
		'''
		bboxes = inp[1][:,:,:5]
		cls_ids = inp[1][:,:,6]
		new_bboxes = inp[2][:,:,:,:5]
		targets = inp[2][:,:,:,-2]
		weights = inp[2][:,:,:,-1]
		'''
		data_time.add(time.time() - start)
		batch_size, num_boxes, num_acts, _ = inp2.shape
		#logger.info(str((inp1.shape, inp2.shape)))

		# forward
		#pred, loss, noweight_loss = model(img_var, bboxes, cls_ids, targets, weights)
		#loss, noweight_loss = loss.mean(), noweight_loss.mean()
		pnum = 100 // num_acts
		iters = math.ceil(num_boxes / pnum)
		#logger.info(iters)
		input_feature = None
		for this_iter in range(iters):
			s = this_iter * pnum
			e = min((this_iter+1) * pnum, num_boxes)

			bboxes = inp1[:,s:e,:5].contiguous().cuda(async=True)
			cls_ids = inp1[:,s:e,6].contiguous().cuda(async=True)
			new_bboxes = inp2[:,s:e,:,:5].contiguous().cuda(async=True)
			targets = inp2[:,s:e,:,-2].contiguous().cuda(async=True)
			weights = inp2[:,s:e,:,-1].contiguous().cuda(async=True)
			
			#forward
			if input_feature is None:
				pred, loss, input_feature = model(img_var, bboxes, new_bboxes, cls_ids, targets, weights)
			else:
				pred, loss = model(img_var, bboxes, new_bboxes, cls_ids, targets, weights, input_feature)
			loss = loss.mean()

			# backward
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			losses.add(loss.item())
			#logger.info('this iter:{}, loss: {}'.format(this_iter, loss.item()))

		#noweight_losses.add(noweight_loss.item())
		batch_time.add(time.time() - start)
		if i % args.log_interval == 0:
			for act_id, act_pred_cuda in enumerate(pred):
				act_pred = act_pred_cuda.cpu().data.numpy()
				logger.info('Act_id {}'.format(act_id)+str((act_pred.min(), act_pred.max(), act_pred.mean())))
			logger.info('Train: [{0}][{1}/{2}]\t'
						'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
						'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
						'Loss {losses.val:.3f} ({losses.avg:.3f})\t'.format(
						#'NWLoss {nwlosses.val:.3f} ({nwlosses.avg:.3f})\t'.format(
							epoch + 1, i, len(train_loader),
							batch_time=batch_time,
							data_time=data_time,
							losses=losses)
							#nwlosses=noweight_losses)
						)
		start = time.time()

if __name__ == "__main__":
	main()
