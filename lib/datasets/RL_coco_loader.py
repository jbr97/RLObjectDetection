import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import logging

class COCODataLoader(DataLoader):
    #TODO
    """

    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False, istrain=True, balancenum=10):
        super(COCODataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                        num_workers, self._collate_fn, pin_memory, drop_last)
        self.istrain = istrain
        self.balancenum = balancenum


    def _collate_fn(self, batch):
        '''
        Return: a mini-batch of data:
            image_data: Variable of image, with shape of [b, 3, max_h, max_w]
            bboxes:     FloatTensor of shape [b, max_num_boxes, 9]  (bid, x1, y1, x2, y2, score, cls_id, image_id, cat_id)
            labels:     FloatTensor of shape [b, max_num_boxes, act_nums, 3] (act_id, label, weight)
            im_infos:   list of len=b, of(
                        resized_image_h, resized_image_w, resize_scale, 
                        origin_image_h, origin_image_w,
                        filename)
        '''
        logger = logging.getLogger('global')
        batch_size = len(batch)

        zip_batch = list(zip(*batch))
        images = zip_batch[0]
        generate_bboxes = zip_batch[1]
        generate_labels = zip_batch[2]
        im_infos = zip_batch[3]





        max_img_h = max([_.shape[-2] for _ in images])
        max_img_w = max([_.shape[-1] for _ in images])

        # for FPN
        #max_img_h = int(np.ceil(max_img_h / 128.0) * 128)
        #max_img_w = int(np.ceil(max_img_w / 128.0) * 128)

        max_num_bboxes = max([_.shape[0] for _ in generate_bboxes])
        num_acts = max([_.shape[1] for _ in generate_labels])
        assert(max_num_bboxes > 0)

        padded_images = torch.FloatTensor(batch_size, 3, max_img_h, max_img_w)
        padded_bboxes = torch.FloatTensor(batch_size, max_num_bboxes, 9)
        padded_labels = torch.FloatTensor(batch_size, max_num_bboxes, num_acts, 3)
        for bid in range(batch_size):
            img = images[bid]
            bboxes = generate_bboxes[bid]
            labels = generate_labels[bid]

            # pad zeros to right bottom of each image
            pad_size = (0, max_img_w - img.shape[-1], 0, max_img_h - img.shape[-2])
            padded_images[bid] = F.pad(img, pad_size).data

            # pad zeros to bboxes
            new_bboxes = torch.FloatTensor(max_num_bboxes, bboxes.shape[-1]).zero_()
            new_bboxes[:bboxes.shape[0]] = bboxes
            batch_id = torch.FloatTensor([bid]).expand((max_num_bboxes, 1))
            padded_bboxes[bid] = torch.cat([batch_id, new_bboxes], dim=1)

            # pad zeros to labels
            new_labels = torch.FloatTensor(max_num_bboxes, labels.shape[-2], labels.shape[-1]).zero_()
            new_labels[:labels.shape[0]] = labels
            padded_labels[bid] = new_labels
        padded_images_var = Variable(padded_images)


        # logger.info('-----------------shape--------------')
        # logger.info('padded images: {}'.format(padded_images_var.shape))
        # logger.info('padded bboxes: {}'.format(padded_bboxes.shape))
        # logger.info('padded_labels: {}'.format(padded_labels.shape))
        # logger.info('-------------------------------------')
        #raise RuntimeError


        
        # padded_labels shape: batch x 100 x 1 x 3, 将其垒起来.
        # padded_bboxes shape: batch x 100 x 8, 将其垒起来.
        # padded images shape: batch x 3 x H x W.
        assert batch_size == padded_bboxes.shape[0] and 100 == padded_bboxes.shape[1], 'Unmatched size: padded_boxes.shape={}'.format(padded_bboxes.shape)
        padded_bboxes = padded_bboxes.view(batch_size * 100, -1)
        padded_labels = padded_labels.view(batch_size * 100, -1)


        if self.istrain:
            ## undersampling均衡不同类别的样本数量。
            ind0 = np.where(padded_labels[:, 1] == 0)[0]
            ind1 = np.where(padded_labels[:, 1] == 1)[0]
            ind2 = np.where(padded_labels[:, 1] == 2)[0]
            ind3 = np.where(padded_labels[:, 1] == 3)[0]
            ind4 = np.where(padded_labels[:, 1] == 4)[0]
            ind5 = np.where(padded_labels[:, 1] == 5)[0]
            ind6 = np.where(padded_labels[:, 1] == 6)[0]

            n_minind = min(len(ind0), len(ind1), len(ind2), len(ind3), len(ind4), len(ind5), len(ind6))

            logger.info('n({}:{}) of different classes: {}, {}, {}, {}, {}, {}, {}'.format(
                        n_minind, self.balancenum,
                        len(ind0), len(ind1), len(ind2), len(ind3), len(ind4), len(ind5), len(ind6)))
            #assert n_minind >= 5, 'The num of samples is too small.'

            n_minind = self.balancenum
            n_pick0 = n_minind if len(ind0) >= n_minind else len(ind0)
            n_pick1 = n_minind if len(ind1) >= n_minind else len(ind1)
            n_pick2 = n_minind if len(ind2) >= n_minind else len(ind2)
            n_pick3 = n_minind if len(ind3) >= n_minind else len(ind3)
            n_pick4 = n_minind if len(ind4) >= n_minind else len(ind4)
            n_pick5 = n_minind if len(ind5) >= n_minind else len(ind5)
            n_pick6 = n_minind if len(ind6) >= n_minind else len(ind6)


            if n_pick0 == 0:
                ind0 = np.array([])
            else:
                ind0 = np.random.choice(ind0, size=n_pick0, replace=False)
            if n_pick1 == 0:
                ind0 = np.array([])
            else:
                ind1 = np.random.choice(ind1, size=n_pick1, replace=False)
            if n_pick2 == 0:
                ind2 = np.array([])
            else:
                ind2 = np.random.choice(ind2, size=n_pick2, replace=False)
            if n_pick3 == 0:
                ind3 = np.array([])
            else:
                ind3 = np.random.choice(ind3, size=n_pick3, replace=False)
            if n_pick4 == 0:
                ind4 = np.array([])
            else:
                ind4 = np.random.choice(ind4, size=n_pick4, replace=False)
            if n_pick5 == 0:
                ind5 = np.array([])
            else:
                ind5 = np.random.choice(ind5, size=n_pick5, replace=False)
            if n_pick6 == 0:
                ind6 == np.array([])
            else:
                ind6 = np.random.choice(ind6, size=n_pick6, replace=False)

            final_ind = np.concatenate([ind0, ind1, ind2, ind3, ind4, ind5, ind6])
            padded_bboxes = padded_bboxes[final_ind, :]
            padded_labels = padded_labels[final_ind, :]

        return [padded_images_var,
                padded_bboxes,
                padded_labels,
                im_infos]

def test(root_dir, ann_file, dt_file):
    import torchvision.transforms as transforms
    from RL_coco_dataset import COCODataset, COCOTransform

    normalize = transforms.Normalize(mean=[0.4485295, 0.4249905, 0.39198247],
                                     std=[0.12032582, 0.12394787, 0.14252729])
    dataset = COCODataset(root_dir, ann_file, dt_file, COCOTransform([800], 1200, flip=False), normalize_fn=normalize)

    loader = COCODataLoader(dataset, batch_size=2, shuffle=False, num_workers=6)
    
    for i, inp in enumerate(loader):
        print('image variable size {} type {}'.format(inp[0].size(), type(inp[0])))
        print('bboxes size {} type {}'.format(inp[1].size(), type(inp[1])))
        print('labels size {} type {}'.format(inp[2].size(), type(inp[2])))
        print('img max_{}, min_{}, mean_{}'.format(inp[0][0].max(), inp[0][0].min(), inp[0][0].mean()))
        print(inp[3])
        print(inp[1][0][0], inp[2][0][0])
        print(inp[1][0][-1], inp[2][0][-1])
        print(inp[1][1][0], inp[2][1][0])
        sys.exit()

if __name__ == "__main__":
    root_dir = "/n/jbr/RL_coco_data/images/val2014"
    ann_file = "/n/jbr/RL_coco_data/annotations/instances_minival2014.json"
    dt_file = "/n/jbr/jsons/detections_minival2014_results.json"
    test(root_dir, ann_file, dt_file)
