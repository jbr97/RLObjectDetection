import numpy as np

def Identify(x):
	return x

class Action:
	def __init__(self, delta, alpha=1., iou_thres=0, wtrans=None):
		self.delta = delta
		self.alpha = alpha
		self.iou_thres = iou_thres
		self.num_acts = 2#4 * len(delta) * 2
		self.actDeltas = np.zeros((self.num_acts, 4), dtype=np.float32)

		self.wtrans = Identify if wtrans is None else wtrans

		self.actDeltas[0, 0] = delta[0] * alpha
		self.actDeltas[1, 0] = -delta[0] * alpha
		'''
		idx = 0
		for i in range(4): # bbox dimention
			for j in range(len(delta)):
				self.actDeltas[idx, i] = delta[j] * alpha
				idx += 1
				self.actDeltas[idx, i] = -delta[j] * alpha
				idx += 1
		'''


	def accuracy_per_box(self, preds, targets):
		"""
			input:
				preds:	np.array of shape b * n * num_acts
				targets:np.array of shape b * n * num_acts
		"""
		batch_size, num_boxes, num_acts = preds.shape
		assert(preds.shape == targets.shape)
		assert(num_acts == self.num_acts)
		
		cnt, correct = 0, 0
		for bid in range(batch_size):
			for i in range(num_boxes):
				act_pred = preds[bid][i]
				act_target = targets[bid][i]

				if act_pred.max() > self.iou_thres:
					cnt += 1
					correct += int(act_target[act_pred.argmax()] == 1)
		cnt = 1 if cnt == 0 else cnt
		return correct * 100. / cnt
		

	def accuracy_per_img(self, preds, targets, maxk=1):
		"""
			input:
				preds:	np.array of shape b * n * num_acts
				targets:np.array of shape b * n * num_acts
				maxk:	int, max number of boxes to be moved
		"""
		batch_size, num_boxes, num_acts = preds.shape
		assert(preds.shape == targets.shape)
		assert(num_acts == self.num_acts)
		
		precs = np.zeros(batch_size, dtype=float)
		for bid in range(batch_size):
			cnt, correct = 0, 0
			vis = [None] * num_boxes
			pred, target = preds[bid], targets[bid]
			inds = np.flip(np.argsort(pred.reshape(-1)), axis=0)
			for num in inds:
				idx = num // self.num_acts
				act_id = num % self.num_acts
				assert(pred.reshape(-1)[num] == pred[idx][act_id])
				if vis[idx] is None:
					cnt += 1
					vis[idx] = 1
					if target[idx][act_id] == 1:
						correct += 1
				if cnt >= maxk:
					break
			precs[bid] = correct * 100. / maxk
		return precs

		
	def accuracy_per_act(self, preds, targets, maxk=None, ratio=.01):
		"""
			input:
				preds:	np.array of shape M * num_acts
				targets:np.array of shape M * num_acts
				maxk:	int, max number of boxes to be moved
				ratio:	float, max ratio number of boxes to be moved
		"""
		num_boxes, num_acts = preds.shape
		assert(preds.shape == targets.shape)
		assert(num_acts == self.num_acts)

		maxk = int(num_boxes * ratio) if maxk is None else maxk
		precs = np.zeros(num_acts, dtype=float)
		preds, targets = preds.transpose(1, 0), targets.transpose(1, 0)
		for act_id in range(num_acts):
			pred, target = preds[act_id], targets[act_id]
			inds = np.argsort(pred)[-maxk:]
			pred, target = pred[inds], target[inds]
			precs[act_id] = (target == 1).sum() * 100. / maxk
		return precs


	def move_from_weight(self, bboxes, weights, targets, maxk=1):
		"""
			input:
				bboxes: np.array of shape b * n * 4
				preds:	np.array of shape b * n * num_acts
				targets:np.array of shape b * n * num_acts
				weights:np.array of shape b * n * num_acts
				maxk:	int, max number of boxes to be moved
		"""
		batch_size, num_boxes, _ = bboxes.shape
		assert(weights.shape == targets.shape)
		assert(bboxes.ndim == 3 and weights.ndim == 3)
		assert(weights.shape[0] == batch_size)
		assert(weights.shape[1] == num_boxes)

		ret = 0
		for bid in range(batch_size):
			cnt = 0
			vis = [None] * num_boxes
			weight, target = weights[bid], targets[bid]
			inds = np.flip(np.argsort(weight.reshape(-1)), axis=0)
			for num in inds:
				idx = num // self.num_acts
				act_id = num % self.num_acts
				assert(weight.reshape(-1)[num] == weight[idx][act_id])
				x, y, w, h =  bboxes[bid][idx]
				delta = self.actDeltas[act_id]
				if vis[idx] is None and target[idx][act_id] == 1:
					ret += weight[idx][act_id]
					cnt += 1
					vis[idx] = 1
					bboxes[bid][idx] += delta * np.array([w,h,w,h])
				if cnt >= maxk:
					break
		return bboxes, float(ret / (maxk * batch_size))

	def move_from_act(self, bboxes, preds, targets, maxk=1):
		"""
			input:
				bboxes: np.array of shape b * n * 4
				preds:	np.array of shape b * n * num_acts
				targets:np.array of shape b * n * num_acts
				maxk:	int, max number of boxes to be moved
		"""
		batch_size, num_boxes, _ = bboxes.shape
		assert(preds.shape == targets.shape)
		assert(bboxes.ndim == 3 and preds.ndim == 3)
		assert(preds.shape[0] == batch_size)
		assert(preds.shape[1] == num_boxes)

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
						bboxes[bid][idx] += delta * np.array([w,h,w,h])
				if cnt >= maxk:
					break
		return bboxes
