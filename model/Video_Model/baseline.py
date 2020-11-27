'''
this model is based on debin meng
the idea is use the frames of video and add the attention weitghs
'''
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from model.Image_Model.resnet2 import ResNet18
import torchvision


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 0.7853975 - 1))
    return norm_angle


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
        self.relu = nn.ReLU()
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
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU()
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

        out = out + residual
        out = self.relu(out)

        return out


###''' self-attention; relation-attention '''

class ResNet_AT(nn.Module):
    def __init__(self, block, layers, num_classes=7, end2end=True, num_pair=1, at_type='self_relation-attention'):
        self.inplanes = 64
        self.num_pair = num_pair
        self.end2end = end2end
        #  print('FAN set:[num_pair:{}][at_type:{}]'.format(num_pair, at_type))
        super(ResNet_AT, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

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
        score_fusion = torch.zeros(size=(img.size(0), 7)).cuda()

        for i in range(self.num_pair):
            f = img[:, :, i, :, :]
            f = self.conv1(f)
            f = self.bn1(f)
            f = self.relu(f)
            f = self.maxpool(f)
            f = self.layer1(f)
            f = self.layer2(f)
            f = self.layer3(f)
            f = self.layer4(f)  # bs,512,4,4
            f = self.avgpool(f)
            f = f.view(-1, 512)
            f = F.dropout(f, 0.5)
            f = self.fc(f)
            f = F.softmax(f, dim=1)
            score_fusion += f

        return score_fusion.div(self.num_pair)


''' self-attention; relation-attention '''


# def Baseline(pretrained=False, **kwargs):
#     # Constructs base a ResNet-18 model.
#     model = ResNet_AT(BasicBlock, [2, 2, 2, 2], **kwargs)
#     return model

class Baseline(nn.Module):
    def __init__(self, pretrain=True, context=False,num_classes=7):
        super(Baseline, self).__init__()
        self.context = context
        self.num_classes=num_classes
        if pretrain:
            self.backbone = ResNet18()
            print('load pretrained baseline')
        else:
            print('load not - pretrained baseline')
            self.backbone = nn.Sequential(
                *list(torchvision.models.resnet18(pretrained=True).children())[:-1],
            )

        if context:
            self.context_bkb = nn.Sequential(
                *list(torchvision.models.resnet18(pretrained=True).children())[:-1],
                nn.BatchNorm2d(512),
                nn.LeakyReLU(),
                nn.Dropout(0.5)
            )
        # for m in self.backbone.parameters():
        #     m.requires_grad=False
        self.pc = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
        )
        # self.backbone =nn.Sequential(
        #     IR_50([112,112]),
        # )
        self.fc = nn.Linear(512, num_classes)
        if self.context:
            self.context_fc = nn.Linear(512, num_classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, face, context=True):
        score_fusion = torch.zeros(size=(face.size(0), self.num_classes)).to(self.device)

        for t in range(face.size(2)):
            f = face[:, :, t, :, :]
            res = self.backbone(f)
            res = self.pc(res)
            # res=F.adaptive_avg_pool2d(res,1)
            res = res.view(face.size(0), -1)
            res = self.fc(res)
            score_fusion += res
        if self.context:
            context_score_fusion = torch.zeros(size=(face.size(0), self.num_classes)).to(self.device)
            for t in range(context.size(2)):
                f = context[:, :, t, :, :]
                res = self.context_bkb(f)
                res = res.view(context.size(0), -1)
                context_score_fusion += self.context_fc(res)
        if self.context:
            return score_fusion.div(face.size(2)) + context_score_fusion.div(context.size(2))
        else:
            return score_fusion.div(face.size(2))


if __name__ == '__main__':
    model = Baseline(context=True).cuda()
    # checkpoint = torch.load('../../pretrained_model/ResNet_pretrained.pth.tar')
    # load_state_dict = {k.replace('module.backbone.', ''): v for k, v in checkpoint['state_dict'].items() if
    #                    k.replace('module.backbone.', '') in model.backbone.state_dict()}
    # print(list(load_state_dict.items())[0])
    # model.backbone.load_state_dict(load_state_dict)
    # print(list(model.backbone.state_dict().items())[0])
    x = torch.rand(size=(16, 3, 3, 112, 112)).cuda()
    context = torch.rand(size=(16, 3, 3, 224, 224)).cuda()
    print(model(x, context).size())
