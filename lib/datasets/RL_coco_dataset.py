from __future__ import division

from datasets.RL_coco import COCO

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import math
import os
import json
from PIL import Image

class COCODataset(Dataset):
    category_to_class = {}
    class_to_category = {}

    @staticmethod
    def get_class(category_id):
        return COCODataset.category_to_class[category_id]

    @staticmethod
    def get_category(class_id):
        return COCODataset.class_to_category[class_id]

    def __init__(self, root_dir, anno_file, labels_file, transform_fn = None, normalize_fn=None):
        self.root_dir = root_dir
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn

        self.coco = COCO(anno_file)

        category_ids = self.coco.cats.keys()

        self.category_to_class = {c: i + 1 for i, c in enumerate(sorted(category_ids))}
        self.class_to_category = {i + 1: c for i, c in enumerate(sorted(category_ids))}

        self.img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))

        self.id2label = {}
        self.id2bbox = {}
        self.id2score = {}
        self.id2cat = {}
        labels = json.load(open(labels_file, "r"))
        for label in labels:
            if label["image_id"] in self.id2label.keys():
                self.id2label[label['image_id']].append(label['dious'])
                self.id2bbox[label['image_id']].append(label['bbox'])
                self.id2score[label['image_id']].append(label['score'])
                self.id2cat[label['image_id']].append(label['category_id'])
            else:
                self.id2label[label['image_id']] = [label['dious']]
                self.id2bbox[label['image_id']] = [label['bbox']]
                self.id2score[label['image_id']] = [label['score']]
                self.id2cat[label['image_id']] = [label['category_id']]

    def __len__(self):
        #return 10
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
        #import random
        #idx = 0
        img_id = self.img_ids[idx]

        meta_img = self.coco.imgs[img_id]
        filename = os.path.join(self.root_dir, meta_img['file_name'])
        image_h, image_w = meta_img['height'], meta_img['width']

        bboxes = []
        pos_weight, neg_weight = 0, 0
        for i, box in enumerate(self.id2bbox[img_id]):
            # bbox category
            cat = self.id2cat[img_id][i]
            # bbox category score(useless here)
            score = self.id2score[img_id][i]
            # added IOU by this action
            weight = math.exp(math.fabs(self.id2label[img_id][i]))
            # is added or not
            if self.id2label[img_id][i] > 0:
                label = 1
                pos_weight += weight
            else:
                label = -1
                neg_weight += weight
            bbox = np.array(self.id2bbox[img_id][i] + [score] + [cat] + [label] + [weight])
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]

            bboxes.append(bbox)
        
        ## Weights Normaliztion
        ## Balance positive and negative samples
        normalized_weights = [ b[7]/pos_weight if b[6]>0 else b[7]/neg_weight for b in bboxes ]
        bboxes = np.array(bboxes, dtype = np.float32)
        ## Balance positive and negative samples
        bboxes[:, 7] = np.array(normalized_weights, dtype=np.float32) * bboxes.shape[0] * 0.5

        img = Image.open(filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        assert(img.size[0]==image_w and img.size[1]==image_h)
        ## transform
        if self.transform_fn:
            resize_scale, img, bboxes = self.transform_fn(img, bboxes)
        else:
            resize_scale = 1
        new_image_w, new_image_h = img.size

        ## to tensor
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        if self.normalize_fn != None:
            img = self.normalize_fn(img)

        return [img.unsqueeze(0),
                [new_image_h, new_image_w, resize_scale, image_h, image_w],
                bboxes,
                filename]

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
