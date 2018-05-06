import sys

import os
import time
import math
import json
import pickle
import numpy as np

import torch
from torch.autograd import Variable

import logging
logger = logging.getLogger("global")

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from model.Reinforcement.Policy import DQN
from model.Reinforcement.utils import AveMeter

class Player(object):
    def __init__(self, config):
        self.config = config
        self.max_epoch = config['max_epoch']
        self.target_network_update_freq = config["target_network_update_freq"]
        self.print_freq = config["print_freq"]
        self.ckpt_freq = config["ckpt_freq"]
        self.log_path = config["log_path"]
        self.batch_size = config["batch_size"]
        self.num_actions = config["num_actions"]
        self.num_rl_steps = config["num_rl_steps"]

        # control sample probablity
        self.epsilon = 0.0
        self.eps_iter = 5000

        # sample parameter
        self.sample_num = config["sample_num"]
        self.sample_ratio = config["sample_ratio"]

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        self.policy = DQN(self.config)
        logger.info("DQN model build done")
        self.policy.init_net()
        logger.info("Init Done")

        self._COCO = COCO(config["ann_file"])

    def train(self, train_dataloader):
        iters = 0
        losses = AveMeter(30)
        batch_time = AveMeter(30)
        data_time = AveMeter(30)

        start = time.time()
        for epoch in range(self.max_epoch):
            for i, inp in enumerate(train_dataloader):
                # suppose we have img, bboxes, gts
                # bboxes:[ids, x1, y1, x2, y2, label]
                # gts: [ids, x1, y1, x2, y2, label]
                data_time.add(time.time() - start)
                imgs = inp[0]
                bboxes = inp[1]
                gts = inp[2]

                for j in range(self.num_rl_steps):
                    # get actions from eval_net
                    actions = self.policy.get_action(imgs, bboxes).tolist()

                    # replace some action in random policy
                    for idx in range(len(actions)):
                        if np.random.uniform() > self.epsilon:
                            actions[idx] = np.random.randint(0, self.num_actions + 1)
                    self.epsilon = iters / self.eps_iter
                    # logger.info(len(actions))

                    # compute iou for epoch bbox before and afer action
                    # we can get delta_iou
                    # bboxes, actions, transform_bboxes, delta_iou
                    transform_bboxes = self._transform(bboxes, actions)
                    old_iou = self._compute_iou(gts, bboxes)
                    # logger.info(len(old_iou))
                    new_iou = self._compute_iou(gts, transform_bboxes)
                    # logger.info(len(new_iou))
                    delta_iou = list(map(lambda x: x[0] - x[1], zip(new_iou, old_iou)))

                    # sample bboxes for a positive and negitive balance
                    bboxes, actions, transform_bboxes, delta_iou = self._sample_bboxes(bboxes, actions, transform_bboxes, delta_iou)
                    # logger.info("bbox shape: {}".format(bboxes.shape))
                    # logger.info("action shape: {}".format(len(actions)))
                    # logger.info("transform_bboxes: {}".format(transform_bboxes.shape))
                    # logger.info("delta_iou shape: {}".format(len(delta_iou)))
                    # logger.info(actions)
                    rewards = self._get_rewards(actions, delta_iou)
                    zero_num = len([u for u in actions if u == 0])
                    logger.info("the num of action0 is {}".format(zero_num))
                    if j == self.num_rl_steps - 1:
                        not_end = 0
                    else:
                        not_end = 1
                    loss = self.policy.learn(imgs, bboxes, actions, transform_bboxes, rewards, not_end)

                    losses.add(np.mean(loss))
                    batch_time.add(time.time() - start)
                    if iters % self.print_freq == 0:
                        logger.info('Train: [{0}][{1}/{2}]\t'
                                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                    'Loss {losses.val:.3f} ({losses.avg:.3f})\t'.format(
                                    epoch + 1, i, len(train_dataloader),
                                    batch_time=batch_time,
                                    data_time=data_time,
                                    losses=losses)
                        )

                    if iters % self.ckpt_freq == 0:
                        state = {
                            'iter': iters,
                            'state_dict': self.policy.eval_net.state_dict()
                        }
                        self._save_model(state)
                        logger.info("Save Checkpoint at {} iters".format(iters))

                    if iters % self.target_network_update_freq == 0:
                        self.policy.update_target_network()
                        logger.info("Update Target Network at {} iters".format(iters))

                    start = time.time()
                    bboxes = transform_bboxes
                    iters += 1

    def eval(self, val_data_loader):
        tot_g_0 = 0
        tot_ge_0 = 0
        tot = 0

        start = time.time()

        all_old_bboxes = list()
        all_new_bboxes = list()
        action_nums = [0] * 25
        iou_nums = [0] * 6
        for i, inp in enumerate(val_data_loader):
            imgs = inp[0]
            bboxes = inp[1]
            gts = inp[2]
            resize_scales = inp[3][:, 2]
            ids = inp[5]

            # get actions
            actions = self.policy.get_action(imgs, bboxes).tolist()
            for action in actions:
                action_nums[action] += 1
            # get old_iou & new_iou
            transform_bboxes = self._transform(bboxes, actions)
            old_iou = self._compute_iou(gts, bboxes)
            new_iou = self._compute_iou(gts, transform_bboxes)

            delta_iou = list(map(lambda x: x[0] - x[1], zip(new_iou, old_iou)))

            iou_nums[0] += len([u for u in delta_iou if u >= 0.1])
            iou_nums[1] += len([u for u in delta_iou if u < 0.1 and u > 0.05])
            iou_nums[2] += len([u for u in delta_iou if u < 0.05 and u >= 0])
            iou_nums[3] += len([u for u in delta_iou if u < 0 and u >= -0.05])
            iou_nums[4] += len([u for u in delta_iou if u < -0.05 and u >= -0.1])
            iou_nums[5] += len([u for u in delta_iou if u < -0.1])

            g_0 = len([u for u in delta_iou if u > 0])
            ge_0 = len([u for u in delta_iou if u >= 0])
            logger.info("Acc(>0): {0} Acc(>=0): {1}"
                        .format(g_0 / len(delta_iou), ge_0 / len(delta_iou)))
            tot_g_0 += g_0
            tot_ge_0 += ge_0
            tot += len(delta_iou)

            for j, (old_bbox, new_bbox) in enumerate(zip(bboxes, transform_bboxes)):
                # bbox = (old_bbox[1:5] / resize_scales[j // 100]).tolist()
                # old_ann = {"image_id": int(ids[int(old_bbox[0])]), "category_id":int(old_bbox[5]), "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], "score": old_bbox[6]}
                bbox = (new_bbox[1:5] / resize_scales[j // 100]).tolist()
                new_ann = {"image_id": int(ids[int(new_bbox[0])]), "category_id":int(new_bbox[5]), "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], "score": new_bbox[6]}
                #print (old_ann)
                # all_old_bboxes.append(old_ann)
                all_new_bboxes.append(new_ann)
            """
            if i % 50 == 0:
                self._save_results(all_old_bboxes, os.path.join(self.log_path, "old_results.json"))
                self._do_detection_eval(os.path.join(self.log_path, "old_results.json"))
            """
        logger.info("Acc(>0): {0} Acc(>=0): {1}"
                    .format(tot_g_0 / tot, tot_ge_0 / tot))
        for idx, action_num in enumerate(action_nums):
            logger.info("the num of action {} is {}".format(idx, action_num))
        # self._save_results(all_old_bboxes, os.path.join(self.log_path, "old_results.json"))
        # self._do_detection_eval(os.path.join(self.log_path, "old_results.json"))
        for iou_num in iou_nums:
            logger.info("rate: {}".format(iou_num / tot))
        self._save_results(all_new_bboxes, os.path.join(self.log_path, "new_results.json"))
        self._do_detection_eval(os.path.join(self.log_path, "new_results.json"))

    def _do_detection_eval(self, res_file):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def _save_results(self, all_bboxes, res_file):
        with open(res_file, "w") as f:
            f.write(json.dumps(all_bboxes))
            #for bbox in all_bboxes:
            #    f.write(json.dumps(bbox) + '\n')

    def _load_results(self, res_file):
        print('loading results from {}\n'.format(res_file))
        return [json.loads(line) for line in open(res_file, 'r')]

    def _save_model(self, model):
        save_path = os.path.join(self.log_path, 'model-{}.pth'.format(model['iter']))
        torch.save(model, save_path)

    def _transform(self, bboxes, actions):
        """
        :param bboxes:
        :param actions:
        :return:
        """
        transform_bboxes = bboxes.copy()
        for i, action in enumerate(actions):
            if action == 1:
                transform_bboxes[i, 1] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.02
            elif action == 2:
                transform_bboxes[i, 1] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.05
            elif action == 3:
                transform_bboxes[i, 1] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.1
            elif action == 4:
                transform_bboxes[i, 2] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.02
            elif action == 5:
                transform_bboxes[i, 2] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.05
            elif action == 6:
                transform_bboxes[i, 2] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.1
            elif action == 7:
                transform_bboxes[i, 3] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.02
            elif action == 8:
                transform_bboxes[i, 3] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.05
            elif action == 9:
                transform_bboxes[i, 3] += (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.1
            elif action == 10:
                transform_bboxes[i, 4] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.02
            elif action == 11:
                transform_bboxes[i, 4] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.05
            elif action == 12:
                transform_bboxes[i, 4] += (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.1
            elif action == 13:
                transform_bboxes[i, 1] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.02
            elif action == 14:
                transform_bboxes[i, 1] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.05
            elif action == 15:
                transform_bboxes[i, 1] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.1
            elif action == 16:
                transform_bboxes[i, 2] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.02
            elif action == 17:
                transform_bboxes[i, 2] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.05
            elif action == 18:
                transform_bboxes[i, 2] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.1
            elif action == 19:
                transform_bboxes[i, 3] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.02
            elif action == 20:
                transform_bboxes[i, 3] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.05
            elif action == 21:
                transform_bboxes[i, 3] -= (transform_bboxes[i, 3] - transform_bboxes[i, 1]) * 0.1
            elif action == 22:
                transform_bboxes[i, 4] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.02
            elif action == 23:
                transform_bboxes[i, 4] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.05
            elif action == 24:
                transform_bboxes[i, 4] -= (transform_bboxes[i, 4] - transform_bboxes[i, 2]) * 0.1
        return transform_bboxes

    def _compute_iou(self, gts, bboxes):
        """
        :param gts: [N, 6] [ids, x1, y1, x2, y2, label]
        :param bboxes: [N, 6] [ids, x1, y1, x2, y2, label]
        :return: [N] iou
        """
        ious = []
        for i in range(self.batch_size):
            gt = gts[gts[:, 0] == i][:, 1:5]
            bbox = bboxes[bboxes[:, 0] == i][:, 1:5]
            iou = np.max(self._bbox_iou_overlaps(bbox, gt), 1).tolist()
            ious.extend(iou)
        return ious


    def _bbox_iou_overlaps(self, b1, b2):
        """
        :param b1: [N, K], K >= 4
        :param b2: [N, K], K >= 4
        :return: intersection-over-union pair-wise.
        """
        area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
        area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
        inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
        inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
        inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
        inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
        inter_h = np.maximum(inter_xmax - inter_xmin, 0)
        inter_w = np.maximum(inter_ymax - inter_ymin, 0)
        inter_area = inter_h * inter_w
        union_area1 = area1.reshape(-1, 1) + area2.reshape(1, -1)
        union_area2 = (union_area1 - inter_area)
        return inter_area / np.maximum(union_area2, 1)

    def _sample_bboxes(self, bboxes, actions, tranform_bboxes, delta_iou):
        """
        sample bboxes for balance
        :param bboxes: [N, 6], batch_ids, x1, y1, x2, y2, score
        :param actions:  [N],
        :param tranform_bboxes: [N, 6], same with bboxes
        :param delta_iou: [N]
        :return: sampled result
        """
        fg_inds = np.where(np.array(delta_iou) >= 0)[0]
        bg_inds = np.where(np.array(delta_iou) < 0)[0]
        # logger.info("fg num: {0} bgnum: {1}".format(len(fg_inds), len(bg_inds)))
        # logger.info("bg num: {}".format(len(bg_inds)))
        fg_num = int(self.sample_num * self.sample_ratio)
        if len(fg_inds) > fg_num:
            fg_inds = fg_inds[np.random.randint(len(fg_inds), size=fg_num)]

        bg_num = self.sample_num - len(fg_inds)
        if len(bg_inds) > bg_num:
            bg_inds = bg_inds[np.random.randint(len(bg_inds), size=bg_num)]
        
        logger.info("fg num: {0} bgnum: {1}".format(len(fg_inds), len(bg_inds)))
        inds = np.array(np.append(fg_inds, bg_inds))
        # logger.info(inds)
        return bboxes[inds, :], np.array(actions)[inds].tolist(), tranform_bboxes[inds, :], np.array(delta_iou)[inds].tolist()

    def _get_rewards(self, actions, delta_iou):
        """
        :param actions: [N]
        :param delta_iou: [N]
        :return: rewards: [N]
        """
        rewards = []
        for i in range(len(actions)):
            if actions[i] == 0:
                rewards.append(0.05)
            else:
                rewards.append(math.tan(delta_iou[i] / 0.14))
        return rewards


