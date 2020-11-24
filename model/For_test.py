'''
this model is based on debin meng
the idea is use the frames of video and add the attention weitghs
'''
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
import cv2
import pdb
from model.Additional_model.model_irse import IR_50

def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 0.7853975 - 1))
    return norm_angle

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ResNet_AT(nn.Module):
    def __init__(self, num_classes=7, end2end=True, num_pair=3, at_type='self_relation-attention'):
        self.inplanes = 64
        self.num_pair = num_pair
        self.end2end = end2end
      #  print('FAN set:[num_pair:{}][at_type:{}]'.format(num_pair, at_type))
        super(ResNet_AT, self).__init__()
        self.backbone=IR_50([112,112])

        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.6)
        self.alpha = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.beta = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.pred_fc1 = nn.Linear(512, num_classes)
        self.pred_fc2 = nn.Linear(1024, num_classes)
        self.at_type = at_type

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def forward(self, img):
        vs = []
        alphas = []
        for i in range(self.num_pair):
            f = img[:, :,i, :, :]
            f=self.backbone(f)
            #print(f.size())
            f = F.adaptive_avg_pool2d(f,1)
            f = f.view(-1, 512)
            vs.append(f)
            alphas.append(self.alpha(self.dropout(f)))
        vs_stack = torch.stack(vs, dim=2)
        alphas_stack = torch.stack(alphas, dim=2)
        if self.at_type == 'self-attention':
            vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
            vm1 = self.dropout(vm1)
            pred_score = self.pred_fc1(vm1)
        elif self.at_type == 'self_relation-attention':
            vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
            betas = []
            for i in range(len(vs)):
                vs[i] = torch.cat([vs[i], vm1], dim=1)
                betas.append(self.beta(self.dropout(vs[i])))

            cascadeVs_stack = torch.stack(vs, dim=2)
            betas_stack = torch.stack(betas, dim=2)
            output = cascadeVs_stack.mul(betas_stack * alphas_stack).sum(2).div((betas_stack * alphas_stack).sum(2))
            output = self.dropout2(output)
            pred_score = self.pred_fc2(output)
        else:
            raise NotImplementedError
        return pred_score


''' self-attention; relation-attention '''



if __name__ == '__main__':
    model = ResNet_AT()
    x = torch.rand(size=(4, 3, 3, 112, 112))
    print(model(x).size())
