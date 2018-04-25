from __future__ import division

from coco import COCO

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import math
import os
import json
from PIL import Image
from collections import defaultdict

class COCODataset(Dataset):
    category_to_class = {}
    class_to_category = {}

    @staticmethod
    def get_class(category_id):
        return COCODataset.category_to_class[category_id]

    @staticmethod
    def get_category(class_id):
        return COCODataset.class_to_category[class_id]

    def __init__(self, root_dir, anno_file, dt_file, transform_fn = None, normalize_fn=None):
        self.root_dir = root_dir
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn

        self.coco = COCO(anno_file)

        category_ids = self.coco.cats.keys()

        self.category_to_class = {c: i + 1 for i, c in enumerate(sorted(category_ids))}
        self.class_to_category = {i + 1: c for i, c in enumerate(sorted(category_ids))}

        self.img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))
        self.annIds = self.coco.getAnnIds(imgIds=self.img_ids)
        self.gt_boxes_list = self.coco.loadAnns(self.annIds)
        self.gt_boxes = defaultdict(list)
        for gt in self.gt_boxes_list:
            self.gt_boxes[gt['image_id']].append(gt)

        ## loading initial detection boxes from json file
        self.dt_boxes_list = json.load(open(dt_file, 'r'))
        self.dt_boxes = defaultdict(list)
        for dt in self.dt_boxes_list:
            self.dt_boxes[dt['image_id']].append(dt)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        '''
        Args: index of data
        Return: a single data:
            image_data: FloatTensor, shape [1, 3, h, w]
            image_info: list of [resized_image_h, resized_image_w, resize_scale, origin_image_h, origin_image_w]
            bboxes: np.array, shape [N, 5] (x1,y1,x2,y2,label)
            filename: str
        Warning:
            we will feed fake ground truthes if None
        '''
        img_id = self.img_ids[idx]

        meta_img = self.coco.imgs[img_id]
        filename = os.path.join(self.root_dir, meta_img['file_name'])
        image_h, image_w = meta_img['height'], meta_img['width']

        bboxes = []
        for dt_box in self.dt_boxes[img_id]:
            bbox = dt_box["bbox"]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            score = dt_box['score']
            cat_id = dt_box['category_id']
            cls_id = self.category_to_class[cat_id]
            bboxes.append(bbox + [cls_id] + [score])
        bboxes = np.array(bboxes, dtype=np.float32)

        gts = []
        for gt_box in self.gt_boxes[img_id]:
            bbox = gt_box["bbox"]
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            # score = gt_box['score']
            cat_id = gt_box['category_id']
            cls_id = self.category_to_class[cat_id]
            gts.append(bbox + [cls_id])
        gts = np.array(gts, dtype=np.float32)

        img = Image.open(filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        assert(img.size[0]==image_w and img.size[1]==image_h)
        ## transform
        if self.transform_fn:
            resize_scale, img, bboxes = self.transform_fn(img, bboxes, gts)
        else:
            resize_scale = 1
        new_image_w, new_image_h = img.size

        ## to tensor
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        if self.normalize_fn != None:
            img = self.normalize_fn(img)

        return [img,
                bboxes,
                gts,
                [new_image_h, new_image_w, resize_scale, image_h, image_w],
                filename]

class COCOTransform(object):
    def __init__(self, sizes, max_size, flip=False):
        if not isinstance(sizes, list):
            sizes = [sizes]
        self.scale_min = min(sizes)
        self.scale_max = max(sizes)
        self.max_size = max_size
        self.flip = flip

    def __call__(self, img, bboxes, gts):
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

        if gts.shape[0] > 0:
            gts[:, :4] *= scale

        return scale, img, bboxes, gts