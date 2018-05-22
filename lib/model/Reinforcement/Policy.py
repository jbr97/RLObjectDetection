import sys
import os

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

from model.Reinforcement.DQLNetwork import resnet101

import logging
logger = logging.getLogger("global")

class DQN(object):
    def __init__(self, config):
        """
        set parameters for training
        :param config:
        """
        self.learning_rate = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config["sample_num"]
        self.gamma = config["gamma"]
        self.pretrain = config["pretrain"]
        self.resume = config["resume"]
        self.class_num = config["num_classes"]
        self.action_num = config["num_actions"]
        self.iters = 0

    def init_net(self):
        """
        initialize two networks for DQL
        :return: None
        """
        self.eval_net = resnet101().cuda()
        self.target_net = resnet101().cuda()

        if self.pretrain != "":
            assert os.path.isfile(self.pretrain), '{} is not a valid file'.format(self.pretrain)
            logger.info("load ckpt from {}".format(self.pretrain))
            checkpoint = torch.load(self.pretrain)
            self.eval_net.load_state_dict(checkpoint, strict=False)
            self.target_net.load_state_dict(checkpoint, strict=False)

        if self.resume != "":
            assert os.path.isfile(self.resume), '{} is not a valid file'.format(self.resume)
            logger.info("resume from {}".format(self.resume))
            checkpoint = torch.load(self.resume)
            # start_epoch = checkpoint['epoch']
            self.eval_net.load_state_dict(checkpoint['state_dict'], strict=True)
            self.target_net.load_state_dict(checkpoint['state_dict'], strict=True)

        self.eval_net.freeze_layer()
        self.target_net.freeze_layer()
        self.loss_func = nn.SmoothL1Loss()


        parameter = []
        for par in self.eval_net.parameters():
            if par.requires_grad == True:
                parameter.append(par)
        self.optimizer = torch.optim.Adam(parameter, lr=self.learning_rate)

    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def get_action(self, imgs, bboxes):
        """
        :param imgs:
        :param bboxes:
        :return:
        """
        batch_size = bboxes.shape[0]
        img = Variable(imgs).cuda()
        classes = Variable(torch.LongTensor(bboxes[:, 7]).contiguous().cuda())
        bboxes = bboxes[:, :5]
        bboxes = self._expand_bbox(bboxes)
        bboxes = Variable(torch.FloatTensor(bboxes).contiguous().cuda()) 
        # bboxes = Variable(torch.FloatTensor(bboxes[:, :5])).contiguous().cuda()
        # classes = Variable(torch.FloatTensor(bboxes[:, 7])).contiguous().cuda()

        values = self.eval_net(img, bboxes)

        values = values.view(batch_size, self.class_num, self.action_num)
        values = values[range(batch_size), classes, :].view(batch_size, self.action_num)
        logger.info("the shape of values is {}".format(values.shape))

        max_vals = torch.max(values, 1)[0].cpu().data.numpy()
        max_inds = torch.max(values, 1)[1].cpu().data.numpy()
        action = []
        for max_val, max_ind in zip(max_vals, max_inds):
            if max_val < 0.5:
                action.append(0)
            else:
                action.append(max_ind)
        action = np.array(action)
        # action = torch.max(values, 1)[1].cpu().data.numpy()
        # logger.info(action)
        # action = action[0]
        return action

    def learn(self, imgs, bboxes, actions, transform_bboxes, rewards, not_end):
        self.iters += 1
        batch_size = bboxes.shape[0]

        # learning rate decay
        self._adjust_learning_rate()

        classes = bboxes[:, 7]
        for cls in classes:
            if 0 <= cls and cls <= 79:
                continue
            else:
                logger.info(cls)
        for i in range(len(actions)):
            actions[i] += classes[i]  * self.action_num

        imgs = Variable(imgs.cuda())
        input_dim = bboxes.shape[0]
        #bboxes = Variable(torch.FloatTensor(bboxes[:, :5]).contiguous().cuda())
        actions = Variable(torch.LongTensor(np.array(actions)).cuda())
        #transform_bboxes = Variable(torch.FloatTensor(transform_bboxes[:, :5]).contiguous().cuda())
        rewards = Variable(torch.FloatTensor(rewards).cuda()).view(input_dim, 1)
        classes = Variable(torch.LongTensor(classes).contiguous().cuda())
        
        bboxes = bboxes[:, :5]
        transform_bboxes = transform_bboxes[:, :5]
        new_bboxes = np.concatenate((bboxes, transform_bboxes), axis=1)
        new_bboxes = new_bboxes.reshape(bboxes.shape[0] * 2, bboxes.shape[1])
        #bboxes = self._expand_bbox(bboxes)
        bboxes = Variable(torch.FloatTensor(new_bboxes).contiguous().cuda())

        Q_output = self.eval_net(imgs, bboxes)
        # logger.info("Shape of Q_output: {}".format(Q_output.shape))
        # logger.info("Shape of actions: {}".format(actions.shape))
        Q_eval = Q_output[range(Q_output.shape[0]), actions].view(input_dim, 1)
        # logger.info("Qeval : {}".format(Q_eval.shape))

        #Q_next = self.target_net(imgs, transform_bboxes).detach()
        #Q_next = Q_next.view(batch_size, self.class_num, self.action_num)
        #Q_next = Q_next[range(batch_size), classes, :].view(batch_size, self.action_num)
        #Q_next_mask = Q_next.max(1)[0].view(input_dim, 1)
        #Q_target = rewards + self.gamma * Q_next_mask * not_end
        Q_target = rewards        

        loss = self.loss_func(Q_eval, Q_target)
        # logger.info("Loss {}".format(loss))
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.cpu().data.numpy()

    def _expand_bbox(self, bbox):
        small_bbox = bbox.copy()
        small_bbox[:, 1] += 0.1 * (bbox[:, 3] - bbox[:, 1])
        small_bbox[:, 2] += 0.1 * (bbox[:, 4] - bbox[:, 2])
        small_bbox[:, 3] -= 0.1 * (bbox[:, 3] - bbox[:, 1])
        small_bbox[:, 4] -= 0.1 * (bbox[:, 4] - bbox[:, 2])

        large_bbox = bbox.copy()
        large_bbox[:, 1] -= 0.1 * (bbox[:, 3] - bbox[:, 1])
        large_bbox[:, 2] -= 0.1 * (bbox[:, 4] - bbox[:, 2])
        large_bbox[:, 3] += 0.1 * (bbox[:, 3] - bbox[:, 1])
        large_bbox[:, 4] += 0.1 * (bbox[:, 4] - bbox[:, 2])

        new_bbox = np.concatenate((small_bbox, bbox, large_bbox), axis=1)
        new_bbox = new_bbox.reshape(bbox.shape[0] * 3, bbox.shape[1])
        return new_bbox

    def _adjust_learning_rate(self):
        if self.iters > 8000:
            lr = self.learning_rate * 0.01
        elif self.iters > 6000:
            lr = self.learning_rate * 0.1
        else:
            lr = self.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return
