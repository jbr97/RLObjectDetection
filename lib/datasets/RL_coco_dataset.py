from __future__ import division

import os
import PIL
import math
import time
import json
import random
import numpy as np
from collections import defaultdict
import logging

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.mask import iou as IoU

from datasets.tools.pnw_static import get_weights_statistics

class COCODataset(Dataset):
	# TODO
	"""
	"""
	def __init__(self, root_dir, ann_file, dt_file, bbox_action, transform_fn=None, normalize_fn=None):
		# TODO
		"""
		"""
		logger = logging.getLogger('global')

		self.root_dir = root_dir
		self.transform_fn = transform_fn
		self.normalize_fn = normalize_fn
		logger.info('Loading annotation files...')
		self.cocoGt = COCO(ann_file)
		self.imgIds = sorted(self.cocoGt.getImgIds())
		self.catIds = sorted(self.cocoGt.getCatIds())
		self.cat2cls = dict([(c, i) for i,c in enumerate(self.catIds)])
		self.cls2cat = dict([(i, c) for i,c in enumerate(self.catIds)])
		
		## get groud-truth boxes
		logger.info('Creating ground-truth bounding boxes...')
		self.annIds = self.cocoGt.getAnnIds(imgIds=self.imgIds, catIds=self.catIds)
		self.gt_boxes_list = self.cocoGt.loadAnns(self.annIds)
		self.gt_boxes = defaultdict(list)
		for gt in self.gt_boxes_list:
			self.gt_boxes[gt['image_id'], gt['category_id']].append(gt)

		## loading initial detection boxes from json file
		logger.info('Loading Detection bounding boxes...')
		self.dt_boxes_list = json.load(open(dt_file, 'r'))
		self.dt_boxes = defaultdict(list)
		for dt in self.dt_boxes_list:
			self.dt_boxes[dt['image_id'], dt['category_id']].append(dt)

		## define bbox actions
		self.bbox_action = bbox_action
		## Prepare the statistics of Delta-IoUs
		logger.info('Preparing statistics of Delta-IoUs (weights)...')

		self.pos_tot, self.neg_tot, \
		self.pos_weights, self.neg_weights = \
			get_weights_statistics(
				self.imgIds, self.catIds,
				self.dt_boxes, self.gt_boxes, self.bbox_action, 
				shuffle=True, maxDets=5000, num_workers=32)

		self.pos_wratio = (self.pos_tot + self.neg_tot) / self.pos_weights / 2.
		self.neg_wratio = (self.pos_tot + self.neg_tot) / self.neg_weights / 2.

		logger.info('weight ratios: '+str((self.pos_wratio, self.neg_wratio)))
		logger.info('COCO datasets created.')


	def __len__(self):
		return len(self.imgIds)


	def __getitem__(self, idx):
		'''
		Args: index of data
		Return: a single data:
			img_data:	FloatTensor, shape [3, h, w]
			bboxes:		FloatTensor, shape [Nr_dts, 6] (x1, y1, x2, y2, score, cls_id, img_id)
			labels:		FloatTensor, shape [Nr_dts, act_nums, 3] (act_id, label, weight)
			im_info:	np.array of
					[resized_image_h, resized_image_w, resize_scale, 
					origin_image_h, origin_image_w, 
					filename]
		Warning:
			we will feed fake ground truthes if None
		'''
		logger = logging.getLogger('global')
		img_id = self.imgIds[idx]

		## get all image infos from cocoGt
		meta_img = self.cocoGt.imgs[img_id]
		filename = os.path.join(self.root_dir, meta_img['file_name'])
		origin_img_h, origin_img_w = meta_img['height'], meta_img['width']
		
		## read image data
		img = PIL.Image.open(filename)
		if img.mode == 'L':
			img = img.convert('RGB')
		## enumerate dt_boxes in image
		generate_bboxes = []
		generate_labels = []
		for cat_id in self.catIds:
			for dt_box in self.dt_boxes[img_id, cat_id]:
				#gtboxes = self.gt_boxes[img_id, cat_id]
				bbox = dt_box['bbox']
				w, h = bbox[2], bbox[3]

				gtboxes = [g['bbox'] for g in self.gt_boxes[img_id, cat_id]]
				iscrowd = [int(g['iscrowd']) for g in self.gt_boxes[img_id, cat_id]]
				if len(gtboxes) == 0:
					gtboxes = [[0,0,0,0]]
					iscrowd = [0]

				origin_ious = IoU([bbox], gtboxes, iscrowd)
				
				generate_label = []
				# 修改这一块。label划分更细致，action只有一个，weight去掉，
				# 吐出的sample在iou>0.5的情况下，使用sampling平衡数量。
				## enumerate actions and apply to the bbox
				# for act_id, act_delta in enumerate(self.bbox_action.actDeltas):
				# 	new_bbox = bbox + act_delta * np.array([w, h, w, h])
				# 	new_ious = IoU([new_bbox], gtboxes, iscrowd)
				# 	delta_iou = new_ious.max() - origin_ious.max()

				# 	if delta_iou > self.bbox_action.iou_thres:
				# 		label = 1
				# 		weight = self.bbox_action.wtrans(delta_iou)
				# 		weight *= self.pos_wratio
				# 	else:
				# 		label = -1
				# 		weight = self.bbox_action.wtrans(delta_iou)
				# 		weight *= self.neg_wratio
				#
				#	 generate_label.append([act_id, label, weight])

				for act_id, act_delta in enumerate(self.bbox_action.actDeltas):
					new_bbox = bbox + act_delta * np.array([w, h, w, h])
					new_ious = IoU([new_bbox], gtboxes, iscrowd)
					delta_iou = new_ious.max() - origin_ious.max()

					if delta_iou > 0.1:
						label = 0
					elif 0.1 >= delta_iou and delta_iou > 0.05:
						label = 1
					elif 0.5 >= delta_iou and delta_iou > 0.0:
						label = 2
					elif delta_iou == 0:
						label = 3
					elif 0 > delta_iou and delta_iou >= -0.05:
						label = 4
					elif -0.05 > delta_iou and delta_iou >= -0.1:
						label = 5
					elif -0.1 > delta_iou:
						label = 6
						
					weight = 1.0
					generate_label.append([act_id, label, weight])

				## attach the generated bbox and labels
				score = dt_box['score']
				cls_id = self.cat2cls[cat_id]
				bbox[2] += bbox[0]
				bbox[3] += bbox[1]
				generate_bboxes.append(bbox+[score]+[cls_id]+[img_id])
				generate_labels.append(generate_label)

		## image data processing
		generate_bboxes = np.array(generate_bboxes)
		generate_labels = np.array(generate_labels)


		## undersampling均衡不同类别的样本数量。
		ind0 = np.where(generate_labels[:, 1] == 0)[0]
		ind1 = np.where(generate_labels[:, 1] == 1)[0]
		ind2 = np.where(generate_labels[:, 1] == 2)[0]
		ind3 = np.where(generate_labels[:, 1] == 3)[0]
		ind4 = np.where(generate_labels[:, 1] == 4)[0]
		ind5 = np.where(generate_labels[:, 1] == 5)[0]
		ind6 = np.where(generate_labels[:, 1] == 6)[0]

		n_minind = min(len(ind0), len(ind1), len(ind2), len(ind3), len(ind4), len(ind5), len(ind6))

		logger.info('n of different classes: {}, {}, {}, {}, {}, {}, {}'.format(
					len(ind0), len(ind1), len(ind2), len(ind3), len(ind4), len(ind5), len(ind6)))
		assert n_minind >= 5, 'The num of samples is too small.'

		ind0 = np.random.choice(ind0, size=n_minind, replace=False)
		ind1 = np.random.choice(ind1, size=n_minind, replace=False)
		ind2 = np.random.choice(ind2, size=n_minind, replace=False)
		ind3 = np.random.choice(ind3, size=n_minind, replace=False)
		ind4 = np.random.choice(ind4, size=n_minind, replace=False)
		ind5 = np.random.choice(ind5, size=n_minind, replace=False)
		ind6 = np.random.choice(ind6, size=n_minind, replace=False)

		final_ind = np.concatenate([ind0, ind1, ind2, ind3, ind4, ind5, ind6])
		generate_bboxes = generate_bboxes[final_ind, :]
		generate_labels = generate_labels[final_ind, :]


		if self.transform_fn:
			# resize_scale, img, bboxes = self.transform_fn(img, generate_bboxes)    # TODO 返回的bboxes没有用过
			resize_scale, img, generate_bboxes = self.transform_fn(img, generate_bboxes)
		else:
			resize_scale = 1
		resize_img_w, resize_img_h = img.size
		to_tensor = transforms.ToTensor()
		img_data = to_tensor(img)
		if self.normalize_fn:
			img_data = self.normalize_fn(img_data)

		## labels to tensor		   
		generate_bboxes = torch.FloatTensor(generate_bboxes)
		generate_labels = torch.FloatTensor(generate_labels)

		## construct im_info
		im_info = [resize_img_h, resize_img_w, resize_scale,
				origin_img_h, origin_img_w,
				filename]

		return [img_data, 
				generate_bboxes, 
				generate_labels,
				im_info]

class COCOTransform(object):
	def __init__(self, sizes, max_size, flip=False):
		if not isinstance(sizes, list):
			sizes = [sizes]
		self.scale_min = min(sizes)
		self.scale_max = max(sizes)
		self.max_size = max_size
		self.flip = flip

	def __call__(self, img, bboxes):
		image_w, image_h = img.size
		short = min(image_w, image_h)
		large = max(image_w, image_h)

		size = np.random.randint(self.scale_min, self.scale_max + 1)
		scale = min(size / short, self.max_size / large)

		new_image_w, new_image_h = math.floor(image_w * scale), math.floor(image_h * scale)

		# new_image_w = new_image_h = self.max_size
		img = img.resize((new_image_w, new_image_h))
		if bboxes.shape[0] > 0:
			bboxes[:, :4] *= scale

		if self.flip:
			if np.random.random() < 0.5:
				img = img.transpose(Image.FLIP_LEFT_RIGHT)

				if bboxes.shape[0] > 0:
					bboxes[:, 0], bboxes[:, 2] = new_image_w - scale - bboxes[:, 2], new_image_w - scale - bboxes[:, 0]

		return scale, img, bboxes
