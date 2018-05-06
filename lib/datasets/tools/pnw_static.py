import math
import random
import numpy as np
from multiprocessing import Process, Queue
from pycocotools.mask import iou as IoU

def get_weights_statistics(
			imgIds, catIds, 
			dt_boxes, gt_boxes, bbox_action, 
			shuffle=False, maxDets=100000, num_workers=2):
	"""
		
	"""
	# TODO
	if shuffle:
		random.shuffle(imgIds)
	if len(imgIds) > maxDets:
		imgIds = imgIds[:maxDets]

	procs = []
	img_start = 0
	L = len(imgIds) // num_workers + 1
	result_queue = Queue(50*num_workers)
	for i in range(num_workers):
		img_end = min(img_start + L, len(imgIds))
		p = Process(target=worker, args=(
			imgIds[img_start:img_end], catIds,
			dt_boxes, gt_boxes, bbox_action, result_queue))
		img_start = img_end
		p.start()
		procs.append(p)

	pos_w, neg_w = 0., 0.
	pos_tot, neg_tot = 0, 0
	for i in range(len(imgIds)):
		pt, nt, pw, nw = result_queue.get()
		pos_tot += pt
		neg_tot += nt
		pos_w += pw
		neg_w += nw

	for p in procs:
		p.join()

	return pos_tot, neg_tot, pos_w, neg_w


def worker(imgIds, catIds, dt_boxes, gt_boxes, bbox_action, result_queue):
	for i, img_id in enumerate(imgIds):
		pt, nt, pw, nw = 0, 0, 0., 0.
		for cat_id in catIds:
			for dt_box in dt_boxes[img_id, cat_id]:
				bbox = dt_box['bbox']
				w, h = bbox[2], bbox[3]

				gtboxes = [g['bbox'] for g in gt_boxes[img_id, cat_id]]
				iscrowd = [int(g['iscrowd']) for g in gt_boxes[img_id, cat_id]]
				if len(gtboxes) == 0:
					gtboxes = [[0,0,0,0]]
					iscrowd = [0]

				origin_ious = IoU([bbox], gtboxes, iscrowd)
				
				## enumerate actions and apply to the bbox
				for act_id, act_delta in enumerate(bbox_action.actDeltas):
					new_bbox = bbox + act_delta * np.array([w, h, w, h])
					new_ious = IoU([new_bbox], gtboxes, iscrowd)
					delta_iou = new_ious.max() - origin_ious.max()

					if delta_iou > bbox_action.iou_thres:
						pt += 1
						weight = bbox_action.wtrans(delta_iou)
						pw += weight
					else:
						nt += 1
						weight = bbox_action.wtrans(delta_iou)
						nw += weight

			'''
			dtboxes = [d['bbox'] for d in dt_boxes[img_id, cat_id]]
			if len(dtboxes) == 0:
				continue
			gtboxes = [g['bbox'] for g in gt_boxes[img_id, cat_id]]
			iscrowd = [int(g['iscrowd']) for g in gt_boxes[img_id, cat_id]]
			if len(gtboxes) == 0:
				gtboxes = [[0,0,0,0]]
				iscrowd = [0]

			# np.array of shape (n, m)
			origin_ious = IoU(dtboxes, gtboxes, iscrowd)
			#inds = origin_ious.argmax(1)
			# np.array of shape (n)
			origin_ious = origin_ious.max(1)
			dtboxes = np.array(dtboxes)
			# np.array of shape (n)
			dt_w, dt_h = dtboxes[:,2], dtboxes[:,3]
			act_alpha = np.array([dt_w, dt_h, dt_w, dt_h]).transpose(1,0)
				
			for act_id, act_delta in enumerate(bbox_action.actDeltas):
				newboxes = dtboxes + act_delta * act_alpha
				new_ious = IoU(newboxes, gtboxes, iscrowd)
				new_ious = new_ious.max(1)
				#mask = np.eye(new_ious.shape[1])[inds]
				#new_ious = (new_ious * mask).sum(1)
				# np.array of shape (n)
				delta_ious = new_ious - origin_ious
				# calculate postives mask
				mask = (delta_ious > bbox_action.iou_thres).astype(int)
				pt += mask.sum()
				nt += (1-mask).sum()
				pos_ious = delta_ious * mask
				neg_ious = delta_ious * (1-mask)
				pw += np.sqrt(np.abs(pos_ious)).sum()
				nw += np.sqrt(np.abs(neg_ious)).sum()
				'''
		result_queue.put((pt, nt, pw, nw))
