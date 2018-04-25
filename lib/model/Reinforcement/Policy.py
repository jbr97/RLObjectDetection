import sys

import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

from DQLNetwork import resnet101


class DQN(object):
    def __init__(self, config):
        """
        set parameters for training
        :param config:
        """
        self.learning_rate = config.lr
        self.batch_size = config.sample_num
        self.gamma = 1.0
        self.iters = 0

    def init_net(self):
        """
        initialize two networks for DQL
        :return: None
        """
        self.eval_net = resnet101().cuda()
        self.target_net = resnet101().cuda()

        self.loss_func = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)

    def update_target_network(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def get_action(self, imgs, bboxes):
        """
        :param imgs:
        :param bboxes:
        :return:
        """
        img = Variable(torch.FloatTensor(imgs).cuda()),
        bboxes = Variable(torch.FloatTensor(bboxes[:5]).contiguous().cuda())

        values = self.eval_net(img, bboxes)

        action = torch.max(values, 1)[1].data.numpy()

        # action = action[0]
        return action

    def learn(self, imgs, bboxes, actions, transform_bboxes, rewards):
        self.iters += 1

        # learning rate decay
        # self._adjust_learning_rate()

        imgs = Variable(torch.FloatTensor(imgs).cuda())
        bboxes = Variable(torch.FloatTensor(bboxes[:5]).contiguous().cuda())
        actions = Variable(torch.LongTensor(np.array(actions)).cuda()).view(self.batch_size, 1)
        transform_bboxes = Variable(torch.FloatTensor(transform_bboxes[:5]).contiguous().cuda())
        rewards = Variable(torch.FloatTensor(rewards).cuda()).view(self.batch_size, 1)

        Q_eval = self.eval_net(imgs, bboxes).gather(1, actions)

        Q_next = self.target_net(imgs, transform_bboxes).detach()
        Q_next_mask = Q_next.max(1)[0].view(self.batch_size, 1)
        Q_target = rewards + self.gamma * Q_next_mask

        loss = self.loss_func(Q_eval, Q_target)

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.data.numpy()

    def _adjust_learning_rate(self):
        if self.iters > 80000:
            lr = self.learning_rate * 0.01
        elif self.iters > 60000:
            lr = self.learning_rate * 0.1
        else:
            lr = self.learning_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return
