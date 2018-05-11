import sys
import os
import time
import math
import json
import pickle
import collections

import numpy as np

import torch
from torch.autograd import Variable

import logging
logger = logging.getLogger("global")

from pycocotools.coco import COCO
from pycocotools.mask import iou as IoU
from pycocotools.cocoeval import COCOeval
from model.Reinforcement.Policy import DQN
from model.Reinforcement.utils import AveMeter, Counter

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

        reward_cnt = Counter(100*self.batch_size)
        diou_cnt = Counter(100*self.batch_size)

        start = time.time()
        for epoch in range(self.max_epoch):
            for i, inp in enumerate(train_dataloader):                                  #　TODO： 是否shuffle？ YES
                # suppose we have img, bboxes, gts
                # bboxes:[batch_id, x1, y1, x2, y2, category, score, cls_id, scale]
                # gts: [batch_id, x1, y1, x2, y2, category, iscrowd, cls_id]
                data_time.add(time.time() - start)
                imgs = inp[0]
                bboxes = inp[1]
                gts = inp[2]

                print('image shape:', imgs.shape)
                print('bboxes shape:', bboxes.shape)
                print('gts shape:', gts.shape)
                # raise RuntimeError

                for j in range(self.num_rl_steps):
                    # get actions from eval_net
                    actions = self.policy.get_action(imgs, bboxes).tolist()

                    # replace some action in random policy
                    for idx in range(len(actions)):
                        # if np.random.uniform() > max(self.epsilon, 0.05):
                        actions[idx] = np.random.randint(0, self.num_actions+1)
                    self.epsilon = iters / self.eps_iter
                    # logger.info(len(actions))

                    # compute iou for epoch bbox before and afer action
                    # we can get delta_iou
                    # bboxes, actions, transform_bboxes, delta_iou
                    transform_bboxes = self._transform(bboxes, actions)                     # TODO: transform换个写法.       DONE
                    old_iou = self._computeIoU(gts, bboxes)                                # TODO: iou 需要考虑到category.   DONE
                    # logger.info(len(old_iou))
                    new_iou = self._computeIoU(gts, transform_bboxes)
                    # logger.info(len(new_iou))
                    delta_iou = list(map(lambda x: x[0] - x[1], zip(new_iou, old_iou)))



                    print('len of actions:', len(actions))
                    print('delta iou lenght:', len(delta_iou))
                    print('leng old iou:', len(new_iou))


                    # sample bboxes for a positive and negitive balance
                    bboxes, actions, transform_bboxes, delta_iou = self._sample_category1_bboxes(bboxes, actions, transform_bboxes, delta_iou)      # TODO: sample 需要换个写法.  加了一个assertion，防止问题。
                    # logger.info("bbox shape: {}".format(bboxes.shape))
                    # logger.info("action shape: {}".format(len(actions)))
                    # logger.info("transform_bboxes: {}".format(transform_bboxes.shape))
                    # logger.info("delta_iou shape: {}".format(len(delta_iou)))
                    # logger.info(actions)
                    rewards = self._get_rewards(actions, delta_iou)                         # TODO: 统计reward的取值分布.   DONE

                    #print('max iou:', max(delta_iou))
                    #print('max reward:', max(rewards))
                    diou_cnt.add(delta_iou)
                    reward_cnt.add(rewards)

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
                                    'Loss {losses.val:.4f} ({losses.avg:.4f})\t'.format(
                                    epoch + 1, i, len(train_dataloader),
                                    batch_time=batch_time,
                                    data_time=data_time,
                                    losses=losses)
                        )
                        a1, a2, a3, a4, a5 = reward_cnt.get_statinfo()
                        d1, d2, d3, d4, d5 = diou_cnt.get_statinfo()
                        logger.info('reward dist: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\tdiou dist: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(a1, a2, a3, a4, a5, 
                                                                                                                                                    d1, d2, d3, d4, d5))

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

    def _get_best_action(self, gts, bboxes):
        actions = []


        diou_cnt = Counter(56)

        for i in range(bboxes.shape[0]):

            bbox = bboxes[i, :][np.newaxis, :]
            max_diou = 0
            best_act = 0
            cnt_eq_iou = 0

            if bbox[0, 5] == 1:
                for act in range(self.num_actions+1):

                    t_bbox = self._transform(bbox, [act])

                    iou1 = self._computeIoU(gts, bbox)
                    iou2 = self._computeIoU(gts, t_bbox)


                    assert len(iou1) == 1 and len(iou2) == 1, 'Unmatched numbers of computeIoU.'

                    iou1 = iou1[0]
                    iou2 = iou2[0]

                    diou_cnt.add(iou2- iou1)

                    if iou1 == iou2:
                        cnt_eq_iou += 1


                    if max_diou < iou2 - iou1:
                        max_diou = iou2 - iou1
                        best_act = act

                
                # print('num of uneffected actions:', cnt_eq_iou)
            else:
                best_act = self.num_actions

            actions.append(best_act)
        return actions
            
    def get_info(self, val_data_loader):

        category_cnt = collections.defaultdict(int)
        all_boxes = list()
        for i, inp in enumerate(val_data_loader):
            bboxes = inp[1]
            resize_scales = inp[3][:, 2]
            ids = inp[5]

            
            for j, bb in enumerate(bboxes):
                # bbox = (old_bbox[1:5] / resize_scales[j // 100]).tolist()
                # old_ann = {"image_id": int(ids[int(old_bbox[0])]), "category_id":int(old_bbox[5]), "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], "score": old_bbox[6]}
                bbox = (bb[1:5] / resize_scales[j // 100]).tolist()
                new_ann = {"image_id": int(ids[int(bb[0])]), "category_id":int(bb[5]), "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], "score": bb[6]}
                #print (old_ann)
                # all_old_bboxes.append(old_ann)
                category_cnt[new_ann["category_id"]] += 1

        most_category_id = None
        max_num = 0
        for k, v in category_cnt.items():
            print('k: {} \t\tv: {}'.format(k, v))

            if v > max_num:
                max_num = v
                most_category_id = k
        
        print('most category id:', most_category_id, 'max_num:', max_num)



    def eval(self, val_data_loader):
        tot_g_0 = 0
        tot_ge_0 = 0
        tot = 0

        start = time.time()

        diou_cnt = Counter(100*self.batch_size)

        all_old_bboxes = list()
        all_new_bboxes = list()
        action_nums = [0] * 57
        iou_nums = [0] * 6
        for i, inp in enumerate(val_data_loader):
            imgs = inp[0]
            bboxes = inp[1]
            gts = inp[2]
            resize_scales = inp[3][:, 2]
            ids = inp[5]

            # get actions
            # actions = self.policy.get_action(imgs, bboxes).tolist()

            # actions = self.policy.get_action_percentage(imgs, bboxes, 0.03).tolist()

            actions = [self.num_actions] * bboxes.shape[0]

            #actions = self._get_best_action(gts, bboxes)


            for action in actions:
                action_nums[action] += 1
            # get old_iou & new_iou
            transform_bboxes = self._transform(bboxes, actions)
            # old_iou = self._compute_iou(gts, bboxes)
            # new_iou = self._compute_iou(gts, transform_bboxes)
            old_iou = self._computeIoU(gts, bboxes)
            new_iou = self._computeIoU(gts, transform_bboxes)


            delta_iou = list(map(lambda x: x[0] - x[1], zip(new_iou, old_iou)))

            diou_cnt.add(delta_iou)

            iou_nums[0] += len([u for u in delta_iou if u >= 0.1])
            iou_nums[1] += len([u for u in delta_iou if u < 0.1 and u > 0.05])
            iou_nums[2] += len([u for u in delta_iou if u < 0.05 and u >= 0])
            iou_nums[3] += len([u for u in delta_iou if u < 0 and u >= -0.05])
            iou_nums[4] += len([u for u in delta_iou if u < -0.05 and u >= -0.1])
            iou_nums[5] += len([u for u in delta_iou if u < -0.1])

            g_0 = len([u for u in delta_iou if u > 0])
            ge_0 = len([u for u in delta_iou if u >= 0])


            d1, d2, d3, d4, d5 = diou_cnt.get_statinfo()    
            logger.info("diou num: {} diou dist: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\t\t Acc(>0): {} Acc(>=0): {}"
                        .format(len(delta_iou), d1, d2, d3, d4, d5, g_0 / len(delta_iou), ge_0 / len(delta_iou)))
            tot_g_0 += g_0
            tot_ge_0 += ge_0
            tot += len(delta_iou)

            for j, (old_bbox, new_bbox) in enumerate(zip(bboxes, transform_bboxes)):
                # bbox = (old_bbox[1:5] / resize_scales[j // 100]).tolist()
                # old_ann = {"image_id": int(ids[int(old_bbox[0])]), "category_id":int(old_bbox[5]), "bbox": [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]], "score": old_bbox[6]}
                # bbox = (new_bbox[1:5] / resize_scales[j // 100]).tolist()
                bbox = (new_bbox[1:5] / new_bbox[8]).tolist()
                
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

        assert bboxes.shape[0] == len(actions), 'Unmatched bboxes and actiosn.'

        transform_bboxes = bboxes.copy()
        for i, action in enumerate(actions):
            if action == 56:
                continue
            else:
                x, y, x2, y2= transform_bboxes[i, 1:5]
                w = x2 - x
                h = y2 - y
                
                # 1-7: [x,y,w,h] -> [x+0.5w, y, w, h]
                if action == 0:   x += w * 0.5**1
                elif action == 1: x += w * 0.5**2
                elif action == 2: x += w * 0.5**3
                elif action == 3: x += w * 0.5**4
                elif action == 4: x += w * 0.5**5
                elif action == 5: x += w * 0.5**6
                elif action == 6: x += w * 0.5**7
                # 8-14: [x,y,w,h] -> [x, y+0.5h, w, h]
                elif action == 7:  y += h * 0.5**1
                elif action == 8:  y += h * 0.5**2
                elif action == 9: y += h * 0.5**3
                elif action == 10: y += h * 0.5**4
                elif action == 11: y += h * 0.5**5
                elif action == 12: y += h * 0.5**6
                elif action == 13: y += h * 0.5**7
                # 15-21: [x,y,w,h] -> [x, y, w+0.5w, h]
                elif action == 14: w += w * 0.5**1
                elif action == 15: w += w * 0.5**2
                elif action == 16: w += w * 0.5**3
                elif action == 17: w += w * 0.5**4
                elif action == 18: w += w * 0.5**5
                elif action == 19: w += w * 0.5**6
                elif action == 20: w += w * 0.5**7
                # 22-28: [x,y,w,h] -> [x, y, w, h+0.5h]
                elif action == 21: h += h * 0.5**1
                elif action == 22: h += h * 0.5**2
                elif action == 23: h += h * 0.5**3
                elif action == 24: h += h * 0.5**4
                elif action == 25: h += h * 0.5**5
                elif action == 26: h += h * 0.5**6
                elif action == 27: h += h * 0.5**7
                # 29-35: [x,y,w,h] -> [x-0.5w, y, w, h]
                elif action == 28: x -= w * 0.5**1
                elif action == 29: x -= w * 0.5**2
                elif action == 30: x -= w * 0.5**3
                elif action == 31: x -= w * 0.5**4
                elif action == 32: x -= w * 0.5**5
                elif action == 33: x -= w * 0.5**6
                elif action == 34: x -= w * 0.5**7
                # 36-42: [x,y,w,h] -> [x, y-0.5h, w, h]
                elif action == 35: y -= h * 0.5**1
                elif action == 36: y -= h * 0.5**2
                elif action == 37: y -= h * 0.5**3
                elif action == 38: y -= h * 0.5**4
                elif action == 39: y -= h * 0.5**5
                elif action == 40: y -= h * 0.5**6
                elif action == 41: y -= h * 0.5**7
                # 43-49: [x,y,w,h] -> [x, y, w-0.5w, h]
                elif action == 42: w -= w * 0.5**1
                elif action == 43: w -= w * 0.5**2
                elif action == 44: w -= w * 0.5**3
                elif action == 45: w -= w * 0.5**4
                elif action == 46: w -= w * 0.5**5
                elif action == 47: w -= w * 0.5**6
                elif action == 48: w -= w * 0.5**7
                # 22,23,24: [x,y,w,h] -> [x, y, w, h-0.5h]
                elif action == 49: h -= h * 0.5**1
                elif action == 50: h -= h * 0.5**2
                elif action == 51: h -= h * 0.5**3
                elif action == 52: h -= h * 0.5**4
                elif action == 53: h -= h * 0.5**5
                elif action == 54: h -= h * 0.5**6
                elif action == 55: h -= h * 0.5**7
                else:
                    raise RuntimeError('Unrecognized action.')

                transform_bboxes[i, 1:5] = np.array([x, y, x+w, y+h])
        return transform_bboxes

    # def _compute_iou(self, gts, bboxes):
    #     """
    #     :param gts: [N, 6] [ids, x1, y1, x2, y2, label]
    #     :param bboxes: [N, 6] [ids, x1, y1, x2, y2, label]
    #     :return: [N] iou
    #     """
    #     ious = []
    #     for i in range(self.batch_size):
    #         gt = gts[gts[:, 0] == i][:, 1:5]
    #         bbox = bboxes[bboxes[:, 0] == i][:, 1:5]
    #         iou = np.max(self._bbox_iou_overlaps(bbox, gt), 1).tolist()
    #         ious.extend(iou)
    #     return ious

    # # add lyj
    # # by jbr   
    # def _computeIoU(self, b, gt_list):
    #     # TODO this function need to be moved
    #     gt = [g['bbox'] for g in gt_list]
    #     iscrowd = [int(g['iscrowd']) for g in gt_list]
    #     if len(gt) == 0:
    #         return 0
    #     ious = IoU([b], gt, iscrowd)

    #     return ious.max()
    # # end lyj


    def _computeIoU(self, gts, bboxes):
        """
        gts: [N, 7], [batch_id, x1, y1, x2, y2, category, iscrowd, cls_id]
        bboxes: [N, 7], [batch_id, x1, y1, x2, y2, category, score, cls_id]
        """


        ious = []
        batch_ids = set(bboxes[:, 0].tolist())

        # for i in range(self.batch_size):
        for i in batch_ids:
            # gt_ind = np.where(gts[:, 0] == i)[0]
            # gt = gts[gt_ind][:, 1:7]
            # dt_ind = np.where(bboxes[:, 0] == i)[0]

            gt = gts[gts[:, 0] == i]
            dt = bboxes[bboxes[:, 0] == i]

            for j in range(dt.shape[0]):
                # get dt bbox.
                dt_bbox = self._transformxywh( dt[j, 1:5] ).tolist()

                # compute category.
                category = dt[j, 5]

                # get gt bbox.
                tmp = gt[gt[:, 5] == category]
                if len(tmp) == 0:
                    gt_bboxes = [[0, 0, 0, 0]]
                    iscrowd = [0]
                else:
                    gt_bboxes = self._transformxywh( tmp[:, 1:5] ).tolist()
                    iscrowd = [int(x) for x in tmp[:, 6]]


                ious.append( IoU(dt_bbox, gt_bboxes, iscrowd).max() )

        return ious


    def _transformxywh(self, bbox):
        if bbox.ndim == 1:
            x1, y1, x2, y2 = bbox
            bounding_boxes = np.array([[ x1, y1, x2-x1, y2-y1 ]])
        elif bbox.ndim == 2:
            n = bbox.shape[0]
            bounding_boxes = np.zeros((n, 4))
            for i in range(n):
                x1, y1, x2, y2 = bbox[i, :]
                bounding_boxes[i, :] = np.array([ x1, y1, x2-x1, y2-y1 ])
        else:
            raise RuntimeError('Unrecognized size of bbox.')

        return bounding_boxes


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
        fg_inds = np.where(np.array(delta_iou) > 0)[0]                                              #  >= changes to >
        bg_inds = np.where(np.array(delta_iou) < 0)[0]
        # logger.info("fg num: {0} bgnum: {1}".format(len(fg_inds), len(bg_inds)))
        # logger.info("bg num: {}".format(len(bg_inds)))
        fg_num = int(self.sample_num * self.sample_ratio)
        bg_num = self.sample_num - fg_num

        if len(fg_inds) < fg_num or len(bg_inds) < bg_num: # TODO: to improve by ratio.
            tmp = min(len(fg_inds), len(bg_inds))
            fg_num = tmp - 1
            bg_num = tmp - 1

        if fg_num <= 0:
            raise RuntimeError('to improve sample bboxes code.')
            
        assert len(fg_inds) >= fg_num and len(bg_inds) >= bg_num, 'sample size is too large.'

        if len(fg_inds) > fg_num:
            fg_inds = fg_inds[np.random.randint(len(fg_inds), size=fg_num)]
            # fg_inds = np.array(delta_iou).argsort()[-fg_num:]

            # fg_iou = np.array(delta_iou)[fg_inds]
            # f1, f2, f3, f4, f5 = self._get_percent_index(fg_inds, fg_iou, fg_num)

        if len(bg_inds) > bg_num:
            bg_inds = bg_inds[np.random.randint(len(bg_inds), size=bg_num)]
            # bg_iou = np.array(delta_iou)[bg_inds]
            # b1, b2, b3, b4, b5 = self._get_percent_index(bg_inds, bg_iou, bg_num)
        
        logger.info("fg num: {0} bgnum: {1}".format(len(fg_inds), len(bg_inds)))
        inds = np.array(np.append(fg_inds, bg_inds))
        # inds = np.array(np.concatenate([f1, f2, f3, f4, f5, b1, b2, b3, b4, b5]))
        # logger.info(inds)

        #print('max iou:', max(np.array(delta_iou)[inds])) 
        return bboxes[inds, :], np.array(actions)[inds].tolist(), tranform_bboxes[inds, :], np.array(delta_iou)[inds].tolist()

    def _sample_category1_bboxes(self, bboxes, actions, transform_bboxes, delta_iou):
        inds = np.where(bboxes[:, 5] == 1)[0]

        num_pos_diou = len([x for x in delta_iou if x > 0])
        num_neg_diou = len([x for x in delta_iou if x < 0])

        logger.info('num of pos diou:{}  num of neg diou:{}'.format(num_pos_diou, num_neg_diou))

        return bboxes[inds, :], np.array(actions)[inds].tolist(), transform_bboxes[inds, :], np.array(delta_iou)[inds].tolist()

    def _get_percent_index(self, fg_inds, fg_iou, fg_num):
        """
        fg_iou 排序，得到最大点max_iou和最小点min_iou，得到中间0.2, 0.4, 0.6, 0.8大小的点对应的index；从相邻两个index中均匀抽取index.
        fg_iou 是不连续的.
        """

        sorted_index = np.argsort(fg_iou)
        sorted_iou = np.sort(fg_iou)

        min_iou = min(sorted_iou)
        max_iou = max(sorted_iou)

        a2 = min_iou + (max_iou - min_iou) * 0.2
        a4 = min_iou + (max_iou - min_iou) * 0.4
        a6 = min_iou + (max_iou - min_iou) * 0.6
        a8 = min_iou + (max_iou - min_iou) * 0.8

        #print('a2:', a2)
        #print('a4:', a4)
        #print('a6:', a6)
        #print('a8:', a8)


        def get_index(a):
            for i in range(len(sorted_iou)):
                if sorted_iou[i] > a:
                    return i


        b2 = get_index(a2)
        b4 = get_index(a4)
        b6 = get_index(a6)
        b8 = get_index(a8)

        avg_num = int(fg_num * 0.2)

        #print('b2:', b2)
        #print('b4:', b4)
        #print('b6:', b6)
        #print('b8:', b8)


        def get_random_ind(b0, b1):
            if b1 - b0 > avg_num:
                c = np.random.choice(range(b0, b1), avg_num, False)
            elif b1 - b0 > 0:
                c = np.array(range(b0, b1))
            else:
                c = []
            return c

        c0 = get_random_ind(0, b2)
        c2 = get_random_ind(b2, b4)
        c4 = get_random_ind(b4, b6)
        c6 = get_random_ind(b6, b8)
        c8 = get_random_ind(b8, len(sorted_iou))

        # c0 = np.random.choice(range(0, b2), avg_num, False)
        # c2 = np.random.choice(range(b2, b4), avg_num, False) 
        # c4 = np.random.choice(range(b4, b6), avg_num, False)
        # c6 = np.random.choice(range(b6, b8), avg_num, False)
        # c8 = np.random.choice(range(b8, len(sorted_iou)), fg_num-4*avg_num, False)


        d0 = np.array(fg_inds)[sorted_index[c0]]
        d2 = np.array(fg_inds)[sorted_index[c2]]
        d4 = np.array(fg_inds)[sorted_index[c4]]
        d6 = np.array(fg_inds)[sorted_index[c6]]
        d8 = np.array(fg_inds)[sorted_index[c8]]
        
        return d0, d2, d4, d6, d8
        

    def _get_rewards(self, actions, delta_iou):
        """
        :param actions: [N]
        :param delta_iou: [N]
        :return: rewards: [N]
        """
        rewards = []
        for i in range(len(actions)):
            if actions[i] == 0:
                rewards.append(0.02)
            else:
                # rewards.append(math.tan(delta_iou[i] / 0.14))
                rewards.append(delta_iou[i])
        return rewards



