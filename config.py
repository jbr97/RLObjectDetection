import os
import json
import numpy as np
import torchvision.transforms as transforms

class Config:

	num_classes = 80
	pretrained_model = 'data/RL_model_dump/pretrained/faster_rcnn_new.pth'
	# save_directory
	save_dir = 'crossentropy.act1.step1.weight_no.panda3/'

	# train settings
	train_img_short = [800]
	train_img_size = 1200
	train_flip = False
	train_max_epoch = 25
	train_lr_decay = [18, 22]
	# train_data_dir = 'data/coco/images/train2014'
	# train_ann_file = 'data/coco/annotations/instances_train2014.json'
	# train_dt_file = 'data/output/detections_train2014_results.json'
	train_data_dir = 'data/coco/images/val2014'
	train_ann_file = 'data/coco/annotations/instances_minival2014.json'
	train_dt_file = 'data/output/detections_minival2014_results.json'

	# normalize transforms
	normalize = transforms.Normalize(mean=[0.4485295, 0.4249905, 0.39198247],
									 std=[0.12032582, 0.12394787, 0.14252729])

	# test settings
	test_img_short = [800]
	test_img_size = 1200
	test_flip = False
	test_data_dir = 'data/coco/images/val2014'
	test_ann_file = 'data/coco/annotations/instances_minival2014.json'
	test_dt_file = 'data/output/detections_minival2014_results.json'

	# SGD settings
	momentum = 0.9
	weight_decay = 0.0001
	learning_rate = 0.001

	# data_loader settings
	num_workers = 6
	data_shuffle = True
	data_pin_memory = True

	# action settings
	# act_delta = [.5, .25, .125, .0625, .03125, .015625, .008]
	act_delta = [.125]
	act_iou_thres = 0

	@staticmethod
	def act_wtrans(x):
		from math import fabs, sqrt, exp
		return exp(fabs(x))

	def __init__(self, phase='train'):
		self.phase = phase
		if phase == 'train':
			self.data_dir = self.train_data_dir
			self.ann_file = self.train_ann_file
			self.dt_file = self.train_dt_file
		else:
			self.data_dir = self.test_data_dir
			self.ann_file = self.test_ann_file
			self.dt_file = self.test_dt_file

