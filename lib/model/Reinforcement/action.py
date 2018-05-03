import numpy as np

def Identify(x):
	return x

class Action:
	def __init__(self, delta, alpha=1., iou_thres=0, wtrans=None):
		self.delta = delta
		self.alpha = alpha
		self.iou_thres = iou_thres
		self.num_acts = 4 * len(delta) * 2
		self.actDeltas = np.zeros((self.num_acts, 4), dtype=np.float32)

		self.wtrans = Identify if wtrans is None else wtrans

		idx = 0
		for i in range(4): # bbox dimention
			for j in range(len(delta)):
				self.actDeltas[idx, i] = delta[j] * alpha
				idx += 1
				self.actDeltas[idx, i] = -delta[j] * alpha
				idx += 1


	def move_from_act(self, bboxes, preds, targets, maxk):
		"""
			input:
				bboxes: np.array of shape b * n * 4
				preds:	np.array of shape b * n * num_acts
				targest:np.array of shape b * n * num_acts
				maxk:	int, max number of boxes to be moved
		"""
		batch_size, num_boxes, _ = bboxes.shape
		assert(preds.shape == targets.shape)
		assert(bboxes.ndim == 3 and preds.ndim == 3)
		assert(preds.shape[0] == batch_size)
		assert(preds.shape[1] == num_boxes)

		correct = 0
		for bid in range(batch_size):
			cnt = 0
			vis = [None] * num_boxes
			pred, target = preds[bid], targets[bid]
			inds = np.flip(np.argsort(pred.reshape(-1)), axis=0)
			for num in inds:
				idx = num // self.num_acts
				act_id = num % self.num_acts
				assert(pred.reshape(-1)[num] == pred[idx][act_id])
				x, y, w, h =  bboxes[bid][idx]
				delta = self.actDeltas[act_id]
				if vis[idx] is None:
					cnt += 1
					vis[idx] = 1
					if target[idx][act_id] == 1:
						correct += 1
						bboxes[bid][idx] += delta * np.array([w,h,w,h])
				if cnt >= maxk:
					break
		return bboxes, correct * 100. / (batch_size * maxk)




		
