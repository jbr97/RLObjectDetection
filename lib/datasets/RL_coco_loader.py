import os
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def to_np_array(x):
    if isinstance(x, Variable): x = x.data
    return x.cpu().numpy() if torch.is_tensor(x) else np.array(x)

class COCODataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False):
        super(COCODataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                        num_workers, self._collate_fn, pin_memory, drop_last)
    def _collate_fn(self, batch):
        '''
        Return: a mini-batch of data:
            image_data: FloatTensor of image, with shape of [b, 3, max_h, max_w]
            image_info: np.array of shape [b, 5], (resized_image_h, resized_image_w, resize_scale, origin_image_h, origin_image_w)
            bboxes: np.array of shape [b, max_num_gts, 5]
            filename: list of str
        '''
        batch_size = len(batch)

        zip_batch = list(zip(*batch))
        images = zip_batch[0]
        unpad_image_sizes = zip_batch[1]
        ground_truth_bboxes = zip_batch[2]
        filenames = zip_batch[3]

        max_img_h = max([_.shape[-2] for _ in images])
        max_img_w = max([_.shape[-1] for _ in images])

        max_img_h = int(np.ceil(max_img_h / 128.0) * 128)
        max_img_w = int(np.ceil(max_img_w / 128.0) * 128)

        max_num_gt_bboxes = max([_.shape[0] for _ in ground_truth_bboxes])
        assert(max_num_gt_bboxes > 0)


        padded_images = []
        padded_gt_bboxes = []
        for b_ix in range(batch_size):
            img = images[b_ix]

            # pad zeros to right bottom of each image
            pad_size = (0, max_img_w - img.shape[-1], 0, max_img_h - img.shape[-2])
            padded_images.append(F.pad(img, pad_size, 'constant', 0).data.cpu())

            # pad zeros to gt_bboxes
            gt_bboxes = to_np_array(ground_truth_bboxes[b_ix])
            new_gt_bboxes = np.zeros([max_num_gt_bboxes, gt_bboxes.shape[-1]])
            new_gt_bboxes[range(gt_bboxes.shape[0]), :] = gt_bboxes
            batch_id = np.zeros([max_num_gt_bboxes, 1])
            batch_id[:, :] = b_ix
            padded_gt_bboxes.extend(np.concatenate((batch_id,new_gt_bboxes), axis=1))

        padded_images = torch.cat(padded_images, dim = 0)
        unpad_image_sizes = np.stack(unpad_image_sizes, axis = 0)
        stack_fn = lambda x : np.stack(x, axis=0) if x else np.array([])
        padded_gt_bboxes = stack_fn(padded_gt_bboxes)

        #logger.debug('image.shape:{}'.format(padded_images.shape))
        #logger.debug('gt_box.shape:{}'.format(padded_gt_bboxes.shape))
        #logger.debug('image_info.shape:{}'.format(unpad_image_sizes.shape))
        #logger.debug('gt_kpts.shape:{}'.format(padded_gt_keypoints.shape))
        #logger.debug('gt_mask.shape:{}'.format(padded_gt_masks.shape))
        return [padded_images,
                unpad_image_sizes,
                padded_gt_bboxes,
                filenames]

def test(root_dir, anno_file, labels_file):
    import torchvision.transforms as transforms
    from coco_dataset import COCODataset, COCOTransform

    normalize = transforms.Normalize(mean=[0.4485295, 0.4249905, 0.39198247],
                                     std=[0.12032582, 0.12394787, 0.14252729])
    dataset = COCODataset(root_dir, anno_file, labels_file, COCOTransform([800], 1200, flip=True), normalize_fn=normalize)

    loader = COCODataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    for iter, input in enumerate(loader):
        batch_size = input[0].shape[0]
        gt_boxes = input[2]
        filenames = input[3]

        print ("the shape of img is {}".format(input[0].shape))
        print ("the shape of gt_boxes is {}".format(gt_boxes.shape))
        print (gt_boxes[0])
        print (gt_boxes[101])
        """
        for b in range(batch_size):
            print ("img name is {}".format(filenames))
            bxs = gt_boxes[b]

            for ix in range(bxs.shape[0]):
                print ("bboxes: {}".format(bxs[ix]))
        """

if __name__ == "__main__":
    root_dir = "/mnt/lustre/share/DSK/datasets/mscoco2017/val2017"
    anno_file = "/mnt/lustre/share/DSK/datasets/mscoco2017/annotations/instances_val2017.json"
    labels_file = "minival2014_action_01.json"
    test(root_dir, anno_file, labels_file)
