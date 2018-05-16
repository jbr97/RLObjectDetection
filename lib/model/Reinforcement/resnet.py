
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.roi_align.modules.roi_align import RoIAlignAvg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
	   'resnet152']


model_urls = {
	'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
	'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
	'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
		   padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
		  residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
					 padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
		  residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, block, layers, num_acts=1, num_classes=1):
		self.inplanes = 64
		self.num_acts = num_acts
		self.num_classes = num_classes
		super(ResNet, self).__init__()

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
					 bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		# it is slightly better whereas slower to set stride = 1
		#self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

		self.RCNN_roi_align = RoIAlignAvg(7, 7, 1.0 / 16.0)

		#self.fc8 = nn.Linear(2048, 4096)
		#self.fc = nn.Linear(2048, num_acts * num_classes)
		self.linears = nn.ModuleList([nn.Linear(4096, num_classes) for i in range(num_acts)])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				#m.weight.data.zero_()
				m.bias.data.zero_()
				m.weight.data.normal_(0, 0.01)

	def freeze_layer(self):
		self._freeze_module(self.conv1)
		self._freeze_module(self.bn1)
		for layer in [self.layer1, self.layer2, self.layer3]:
			self._freeze_module(layer)

	def _freeze_module(self, module):
		for p in module.parameters():
			p.requires_grad = False

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
			nn.Conv2d(self.inplanes, planes * block.expansion,
				  kernel_size=1, stride=stride, bias=False),
			nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, img, bboxes, new_bboxes, cls_ids, targets, weights, input_feature=None):
		# b * nb * na * 5 -> na * b * nb * 5
		new_bboxes = new_bboxes.transpose(0,2).transpose(1,2).contiguous().view(-1,5)
		#bboxes = bboxes.expand_as(new_bboxes).view(-1, 5)
		# b * nb * 5
		bboxes = bboxes.view(-1, 5)
		#new_bboxes = new_bboxes.view(-1, 5)
		batch_size = bboxes.size(0)

		cls_ids = cls_ids.view(batch_size)
		targets = targets.view(batch_size, self.num_acts)
		weights = weights.view(batch_size, self.num_acts)

		if input_feature is None:
			x = self.conv1(img)
			x = self.bn1(x)
			x = self.relu(x)
			x = self.maxpool(x)

			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
			ret_feat = x
		else:
			x = input_feature
		
		cat_boxes = torch.cat((bboxes, new_bboxes), 0)
		# feature: (na * b * nb, 1024, 7, 7)
		x = self.RCNN_roi_align(x, cat_boxes)
		'''
		# feature: (na * b * nb, 1024, 7, 7)
		act_feat = self.RCNN_roi_align(x, new_bboxes)
		# feature: (b * nb, 1024, 7, 7)
		box_feat = self.RCNN_roi_align(x, bboxes)
		# feature: ((1+na) * b * nb, 1024, 7, 7)
		x = torch.cat((box_feat, act_feat), 0)
		'''

		# head to tail
		'''
		feat = self.layer4(x[:batch_size])
		feat = feat.mean(3)
		feat = feat.mean(2)
		for i in range(self.num_acts):
			s, e = batch_size * (i+1), batch_size * (i+2)
			tmp = self.layer4(x[s:e])
			tmp = tmp.mean(3)
			tmp = tmp.mean(2)
			feat = torch.cat((feat, tmp), 0)
		x = feat
		'''
		x = self.layer4(x)
		# avg pool
		x = x.mean(3)
		x = x.mean(2)
		
		#pred = self.fc(x)
		# with shape (batch*num_classes, num_acts)
		#pred = pred.reshape(-1, self.num_acts)
		cls_ids += torch.range(0, self.num_classes * batch_size - 1, self.num_classes).cuda()
		# index select with class indices
		cls_ids = cls_ids.type(torch.cuda.LongTensor)
		# feature: ((1+na) * b * nb, 2048) -> (1+na, b*nb, 2048)
		x = x.view(-1, batch_size, 2048)

		loss = None
		pred = []
		targets = targets.transpose(1, 0)
		weights = weights.transpose(1, 0)
		for i, fc in enumerate(self.linears):
			# 2 * (b*nb, 2048) -> (b*nb, 4096)
			in_feat = torch.cat((x[0], x[i+1]), 1)
			# pred: (b*nb, num_classes) -> (batch_size * num_classes)
			temp_pred = fc(in_feat).reshape(-1)
			pred.append(temp_pred.index_select(0, cls_ids))
			if loss is None:
				loss = self._weighted_mse_loss(pred[i], targets[i], weights[i])
			else:
				loss += self._weighted_mse_loss(pred[i], targets[i], weights[i])
		loss /= self.num_acts
		
		#pred = torch.index_select(pred, 0, cls_ids)
		# pred, targets, weights: (batch, num_acts)
		#loss, noweight_loss = self._weighted_mse_loss(pred, targets, weights)
		if input_feature is None:
			return pred, loss, ret_feat
		else:
			return pred, loss#, noweight_loss

	def _weighted_mse_loss(self, inp, targets, weights):
		# TODO move this function
		noweight_loss = (inp - targets) ** 2
		loss = noweight_loss * (weights.expand_as(noweight_loss))
		return loss.mean()#, noweight_loss.mean()

def resnet18(pretrained=False):
	"""Constructs a ResNet-18 model.
	Args:
	pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [2, 2, 2, 2])
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
	return model


def resnet34(pretrained=False):
	"""Constructs a ResNet-34 model.
	Args:
	pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(BasicBlock, [3, 4, 6, 3])
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
	return model


def resnet50(pretrained=False):
	"""Constructs a ResNet-50 model.
	Args:
	pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 6, 3])
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
	return model


def resnet101(pretrained=False, num_acts=1, num_classes=1):
	"""Constructs a ResNet-101 model.
	Args:
	pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 4, 23, 3], 
		num_acts=num_acts, 
		num_classes=num_classes)
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
	return model


def resnet152(pretrained=False):
	"""Constructs a ResNet-152 model.
	Args:
	pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = ResNet(Bottleneck, [3, 8, 36, 3])
	if pretrained:
		model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
	return model
