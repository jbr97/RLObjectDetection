import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

class COCODataLoader(DataLoader):
	#TODO
	"""

	"""
	def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
				 num_workers=0, pin_memory=False, drop_last=False):
		super(COCODataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
										num_workers, self._collate_fn, pin_memory, drop_last)


	def _collate_fn(self, batch):
		'''
		Return: a mini-batch of data:
			image_data: Variable of image, with shape of [b, 3, max_h, max_w]
			bboxes:		FloatTensor of shape [b, max_num_boxes, 8]	(bid, x1, y1, x2, y2, score, cls_id, img_id)
			labels:		FloatTensor of shape [b, max_num_boxes, act_nums, 7] (bid, x1, y1, x2, y2, label, weight)
			im_infos:	list of len=b, of(
						resized_image_h, resized_image_w, resize_scale, 
						origin_image_h, origin_image_w,
						filename)
		'''
		batch_size = len(batch)

		zip_batch = list(zip(*batch))
		images = zip_batch[0]
		generate_bboxes = zip_batch[1]
		generate_labels = zip_batch[2]
		im_infos = zip_batch[3]

		max_img_h = max([_.shape[-2] for _ in images])
		max_img_w = max([_.shape[-1] for _ in images])

		# for FPN
		max_img_h = int(np.ceil(max_img_h / 128.0) * 128)
		max_img_w = int(np.ceil(max_img_w / 128.0) * 128)

		max_num_bboxes = max([_.shape[0] for _ in generate_bboxes])
		num_acts = max([_.shape[1] for _ in generate_labels])
		assert(max_num_bboxes > 0)

		padded_images = torch.FloatTensor(batch_size, 3, max_img_h, max_img_w)
		padded_bboxes = torch.FloatTensor(batch_size, max_num_bboxes, 8)
		padded_labels = torch.FloatTensor(batch_size, max_num_bboxes, num_acts, 7)
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
			new_labels = torch.FloatTensor(max_num_bboxes, num_acts, labels.shape[-1]).zero_()
			new_labels[:labels.shape[0]] = labels
			batch_id = torch.FloatTensor([bid]).expand((max_num_bboxes, num_acts, 1))
			padded_labels[bid] = torch.cat([batch_id, new_labels], dim=2)
		padded_images_var = Variable(padded_images)

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
