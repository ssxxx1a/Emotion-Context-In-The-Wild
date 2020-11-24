import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import cv2


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride, downsample, temp_conv, temp_stride, use_nl=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1 + temp_conv * 2, 1, 1), stride=(temp_stride, 1, 1),
                               padding=(temp_conv, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bb = nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-2])

    # print(self.bb)
    def forward(self, x):
        x = self.bb[0](x)
        x = self.bb[1](x)
        #  x = self.bb[2](x)
        #  x = self.bb[3](x)
        # x = self.bb[4](x)
        #  x = self.bb[5](x)
        feature_1 = x[0]
        #  print(feature_1.size())
        return x, feature_1


class replace_c3d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0)):
        super(replace_c3d, self).__init__()
        self.v = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=(1, 1, kernel_size[1]),
                      stride=(1, 1, stride[1]),
                      padding=(0, 0, padding[1])),
            nn.Conv3d(in_channels=out_c, out_channels=in_c, kernel_size=(1, 1, 1), stride=1,
                      padding=0)
        )
        self.h = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=(1, kernel_size[2], 1),
                      stride=(1, stride[2], 1),
                      padding=(0, padding[2], 0)),
            nn.Conv3d(in_channels=out_c, out_channels=in_c, kernel_size=(1, 1, 1), stride=1,
                      padding=(0))
        )
        self.t = nn.Conv3d(in_c, out_c, kernel_size=(kernel_size[0], 1, 1), stride=(stride[0], 1, 1),
                           padding=(padding[0], 0, 0))

    def forward(self, x):
        x = self.v(x)
        x = self.h(x)
        x = self.t(x)
        return x


class VH_block(nn.Module):
    def __init__(self, in_c=512, out_c=512, t=2, h_size=7, v_size=7, times=1):
        super(VH_block, self).__init__()
        '''
        1、let h_size and v_size as the size of img //16 respectively
        2、after the conv_v and conv_h, should i need to use matrix mul?
        3、how to choose the stride and padding?
        4、should i fuse the feature map after use multiple conv_v and conv_h respectively?
        '''
        self.Conv_h = nn.Conv3d(in_c, out_c, kernel_size=(t, 1 * times, h_size), stride=(2, 1, 2), padding=(0, 0, 1))
        self.Conv_v = nn.Conv3d(in_c, out_c, kernel_size=(t, v_size, 1 * times), stride=(2, 2, 1), padding=(0, 1, 0))

    def forward(self, x):
        h = self.Conv_h(x)
        v = self.Conv_v(x)
        _v = v.permute(0, 1, 2, 4, 3)
        _h = h.permute(0, 1, 2, 4, 3)
        print(_v.shape)
        print(_h.shape)
        f_h = self.Conv_v(torch.cat([h, _v], dim=2))
        f_v = self.Conv_h(torch.cat([v, _h], dim=2))
        print('after conv_h:', h.size(), 'after conv_v:', v.size())
        print('after fuse feature map:', torch.matmul(h, v).size())
        # h_v = self.Conv_v(self.Conv_h(x))
        # print('after h and v directly:', h_v.size())
        res = torch.cat([f_h, f_v], dim=2)
        res = F.max_pool3d(res, kernel_size=(2, 1, 1))
        return res


class Motion_Net(nn.Module):
    def __init__(self):
        super(Motion_Net, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    x = torch.rand(size=(1, 3, 16, 224, 224))
    # model = VH_block(3, 64)
    c3d = nn.Conv3d(3, 63, 3, 2, 1)
    print('Total params: %.2fM' % (sum(p.numel() for p in c3d.parameters()) / 10000.0))
    # print(model(x).size())
    xx = replace_c3d(3, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
    print('Total params: %.2fM' % (sum(p.numel() for p in xx.parameters()) / 10000.0))
    print(xx(x).size())

# x = torch.rand(size=(1, 3, 224, 224))
# img1 = Image.open('../000015.jpg')
# img2 = Image.open('../000016.jpg')
# t = transforms.Compose([
#     transforms.Resize((224, 224), Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
# ])
# x1 = t(img1).unsqueeze(0)
# x2 = t(img2).unsqueeze(0)
# model = Net()
# res1, f1 = model(x1)
# res2, f2 = model(x2)
# #  f1=f1.numpy()[0]
# print(f1.shape)
# plt.figure(1)
# plt.subplot(321)
# plt.imshow(f1[0].detach().numpy())
# plt.subplot(322)
# plt.imshow(f1[-1].detach().numpy())
# plt.subplot(323)
# plt.imshow(f2[0].detach().numpy())
# plt.subplot(324)
# plt.imshow(f2[-1].detach().numpy())
# plt.subplot(325)
# plt.imshow(torch.sum(f1, 0).detach().numpy())
# plt.subplot(326)
# plt.imshow(torch.sum(f2, 0).detach().numpy())
# plt.show()


# plt.figure(2)
# plt.subplot(211)
# cat_data = torch.cat([f1, f2], dim=0).unsqueeze(0)
# print(cat_data.size())
# dddd = F.max_pool3d(cat_data, kernel_size=(2, 2, 2))
# dddd = dddd.squeeze(0)
# plt.imshow(torch.sum(dddd, 0).detach().numpy())
# plt.subplot(212)
# xxxx = torch.cat([f1[0].unsqueeze(0), f2[0].unsqueeze(0)], dim=0).unsqueeze(0)
# print(xxxx.size())
# xxxx = F.max_pool3d(xxxx, kernel_size=(2, 2, 2))
# print(xxxx.size())
# plt.imshow(xxxx[0][0].detach().numpy())
# plt.show()
