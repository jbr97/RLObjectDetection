from __future__ import division

import os
import PIL
import math
import json
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.mask import iou as IoU
# TODO from models.Reinforcement.RL_actions import action_generator

class COCODataset(Dataset):
    # TODO
    """
    """
    def __init__(self, root_dir, ann_file, dt_file, transform_fn = None, normalize_fn=None):
        # TODO
        """
        """
        self.root_dir = root_dir
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.cocoGt = COCO(ann_file)
        self.imgIds = sorted(self.cocoGt.getImgIds())
        self.catIds = sorted(self.cocoGt.getCatIds())
        self.cat2cls = dict([(c, i) for i,c in enumerate(self.catIds)])
        self.cls2cat = dict([(i, c) for i,c in enumerate(self.catIds)])
        
        ## get groud-truth boxes
        self.annIds = self.cocoGt.getAnnIds(imgIds=self.imgIds, catIds=self.catIds)
        self.gt_boxes_list = self.cocoGt.loadAnns(self.annIds)
        self.gt_boxes = defaultdict(list)
        for gt in self.gt_boxes_list:
            self.gt_boxes[gt['image_id'], gt['category_id']].append(gt)

        ## loading initial detection boxes from json file
        self.dt_boxes_list = json.load(open(dt_file, 'r'))
        self.dt_boxes = defaultdict(list)
        for dt in self.dt_boxes_list:
            self.dt_boxes[dt['image_id'], dt['category_id']].append(dt)

        ## TODO this procedure need to be moved
        boxdim = 4
        alpha = .1
        delta = [1., .5, .1]
        numacts = len(delta) * boxdim * 2
        self.actIds = np.array(range(numacts))
        self.actDeltas = np.zeros((numacts, boxdim), dtype=np.float32)
        num = 0
        for i in range(boxdim):
            for j in range(len(delta)):
                self.actDeltas[num][i] = delta[j] * alpha
                num += 1
                self.actDeltas[num][i] = -delta[j] * alpha
                num += 1


    def __len__(self):
        return len(self.imgIds)


    def _computeIoU(self, b, gt_list):
        # TODO this function need to be moved
        gt = [g['bbox'] for g in gt_list]
        iscrowd = [int(g['iscrowd']) for g in gt_list]
        if len(gt) == 0:
            return 0
        ious = IoU([b], gt, iscrowd)

        return ious.max()
            

    def __getitem__(self, idx):
        '''
        Args: index of data
        Return: a single data:
            img_data:   FloatTensor, shape [3, h, w]
            bboxes:     FloatTensor, shape [Nr_dts, 6] (x1, y1, x2, y2, score, cls_id)
            labels:     FloatTensor, shape [Nr_dts, act_nums, 3] (act_id, label, weight)
            im_info:    np.array of
                    [resized_image_h, resized_image_w, resize_scale, 
                    origin_image_h, origin_image_w, 
                    filename]
        Warning:
            we will feed fake ground truthes if None
        '''
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
                bbox = dt_box['bbox']
                x, y, w, h = bbox
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                score = dt_box['score']
                cls_id = self.cat2cls[cat_id]
                generate_bboxes.append(bbox+[score]+[cls_id])
                # TODO this function need to be moved
                origin_iou = self._computeIoU(bbox, self.gt_boxes[img_id, cat_id])
                
                generate_label = []
                ## enumerate actions and apply to the bbox
                for act_id in self.actIds:
                    delta = self.actDeltas[act_id]
                    new_bbox = bbox + delta * np.array([w, h, w, h])
                    # TODO this function need to be moved
                    new_iou = self._computeIoU(new_bbox, self.gt_boxes[img_id, cat_id])
                    dious = new_iou - origin_iou
                    if dious > 0:
                        label = 1
                    else:
                        label = -1
                    #label = int(dious > 0) * 2 - 1
                    weight = math.exp(math.fabs(dious))
                    generate_label.append([act_id, label, weight])
                generate_labels.append(generate_label)

        ## image data processing
        generate_bboxes = np.array(generate_bboxes)
        generate_labels = np.array(generate_labels)
        if self.transform_fn:
            resize_scale, img, bboxes = self.transform_fn(img, generate_bboxes)
        else:
            resize_scale = 1
        resize_img_w, resize_img_h = img.size
        to_tensor = transforms.ToTensor()
        img_data = to_tensor(img)
        if self.normalize_fn:
            img_data = self.normalize_fn(img_data)

        ## for each action, normalize the weights
        # TODO need a function
        num_boxes = generate_labels.shape[0]

        for act_id in self.actIds: 
            pos_cnt, neg_cnt = 0, 0
            pos_weight, neg_weight = 0, 0
            for i in range(num_boxes):
                # positive samples
                if generate_labels[i][act_id][1] == 1:
                    pos_cnt += 1
                    pos_weight += generate_labels[i][act_id][2]
                else:
                    neg_cnt += 1
                    neg_weight += generate_labels[i][act_id][2]
            for i in range(num_boxes):
                if generate_labels[i][act_id][1] == 1:
                    generate_labels[i][act_id][2] *= (pos_cnt+neg_cnt) / pos_weight / 2
                else:
                    generate_labels[i][act_id][2] *= (pos_cnt+neg_cnt) / neg_weight / 2

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
