# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class custom_Conv_Net(nn.Module):
    def __init__(self):
        super(custom_Conv_Net, self).__init__()
        # self.bias=Parameter(torch.Tensor(1))
        self.encode_img = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.encode_face = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

    def forward(self, img, face):
        # the bs and channels must be equal.
        img = self.encode_img(img)
        face = self.encode_face(face)
        vs = []
        bs = img.size(0)
        for i in range(bs):
            _img = img[i].unsqueeze(dim=0)
            _face = face[i].unsqueeze(dim=0)
            res = F.conv2d(input=_img, weight=_face, stride=1)
            vs.append(res)
        res = torch.stack(vs, dim=0)
        res = res.permute(1, 0, 2, 3, 4).squeeze(dim=0)
       # bs, c, h, w = res.size()
        #  res=res.view(bs,c,-1)
        #  res=F.softmax(res,dim=-1)
        res = F.softmax(res, dim=2)
        res = F.softmax(res, dim=3)
      #  res = res.view(bs, c, h, w)
        return res


if __name__ == '__main__':
    face = torch.rand(size=(17, 512, 7, 7))
    img = torch.rand(size=(17, 512, 14, 14))
    net = custom_Conv_Net()
    _img = img[0].unsqueeze(dim=0)
    _face = face[0].unsqueeze(dim=0)
    print(net(_img, _face))
