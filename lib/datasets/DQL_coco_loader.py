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
        predict_bboxes = zip_batch[1]
        ground_truth_bboxes = zip_batch[2]
        unpad_image_sizes = zip_batch[3]
        filenames = zip_batch[4]

        max_img_h = max([_.shape[-2] for _ in images])
        max_img_w = max([_.shape[-1] for _ in images])

        max_img_h = int(np.ceil(max_img_h / 128.0) * 128)
        max_img_w = int(np.ceil(max_img_w / 128.0) * 128)

        max_num_predict_bboxes = max([_.shape[0] for _ in predict_bboxes])
        assert (max_num_predict_bboxes > 0)
        max_num_gt_bboxes = max([_.shape[0] for _ in ground_truth_bboxes])
        assert  (max_num_gt_bboxes > 0)

        padded_images = []
        padded_predict_bboxes = []
        padded_gt_bboxes = []
        for b_ix in range(batch_size):
            img = images[b_ix]

            # pad zeros to right bottom of each image
            pad_size = (0, max_img_w - img.shape[-1], 0, max_img_h - img.shape[-2])
            padded_images.append(F.pad(img, pad_size, 'constant', 0).data.cpu())

            # pad zero to predict_bboxes
            predict_bboxes = to_np_array(predict_bboxes[b_ix])
            new_predict_bboxes = np.zeros([max_num_predict_bboxes, predict_bboxes.shape[-1]])
            new_predict_bboxes[range(predict_bboxes.shape[0]), :] = predict_bboxes
            batch_id = np.zeros([max_num_predict_bboxes, 1])
            padded_predict_bboxes.extend(np.concatenate((batch_id, predict_bboxes), axis=1))

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
        padded_predict_bboxes = stack_fn(padded_predict_bboxes)
        padded_gt_bboxes = stack_fn(padded_gt_bboxes)

        #logger.debug('image.shape:{}'.format(padded_images.shape))
        #logger.debug('gt_box.shape:{}'.format(padded_gt_bboxes.shape))
        #logger.debug('image_info.shape:{}'.format(unpad_image_sizes.shape))
        #logger.debug('gt_kpts.shape:{}'.format(padded_gt_keypoints.shape))
        #logger.debug('gt_mask.shape:{}'.format(padded_gt_masks.shape))
        return [padded_images,
                padded_predict_bboxes,
                padded_gt_bboxes,
                unpad_image_sizes,
                filenames]