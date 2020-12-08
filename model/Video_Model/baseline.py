'''
this model is based on debin meng
the idea is use the frames of video and add the attention weitghs
'''
import torch.nn as nn
import math
import torch
import time
import torch.nn.functional as F
from model.Image_Model.resnet2 import resnet18
import torchvision
from torchvision import transforms
from utils.util import unnorm_save
class Baseline(nn.Module):
    def __init__(self, pretrain=True, context=False, num_classes=7, radio=1):
        super(Baseline, self).__init__()
        self.context = context
        self.num_classes = num_classes
        if pretrain:
            self.backbone = resnet18()
        else:
            self.backbone = resnet18(radio=radio)
            #  self.backbone = nn.Sequential(
            #      *list(torchvision.models.resnet18(pretrained=False).children())[:-1],
            #      # *list(torchvision.models.resnet34(pretrained=True).children())[:-1],
            #  )
        if context:
            self.context_bkb = nn.Sequential(
                # *list(torchvision.models.resnet34(pretrained=True).children())[:-1],
                *list(torchvision.models.resnet18(pretrained=False).children())[:-1],
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
        self.fc = nn.Linear(int(radio * 512), num_classes)
        if self.context:
            self.context_fc = nn.Linear(512, num_classes)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, face, context=None):
        score_fusion = torch.zeros(size=(face.size(0), self.num_classes)).to(self.device)

        for t in range(face.size(2)):
            f = face[:, :, t, :, :]
            #  transforms.ToPILImage()(f[0].cpu()).save('./f.png')
            res = self.backbone(f)
            res = res.view(face.size(0), -1)
            res = F.softmax(self.fc(res),dim=1)
            score_fusion += res
        if self.context:
            context_score_fusion = torch.zeros(size=(context.size(0), self.num_classes)).to(self.device)
            for t in range(context.size(2)):
                c = context[:, :, t, :, :]
                #     transforms.ToPILImage()(c[0].cpu()).save('./c.png')
                c_res = self.context_bkb(c)
                c_res = c_res.view(context.size(0), -1)
                c_res=F.softmax(self.context_fc(c_res), dim=1)

                context_score_fusion += c_res
        res_f=score_fusion.div(face.size(2))
        if self.context:
            res_c = context_score_fusion.div(context.size(2))

        # xc = torch.max(res_c, 1)[1][0]
        # xf = torch.max(res_f, 1)[1][0]
        # if xc.item()==xf.item():
        #     if res_c[0][xc]>res_f[0][xf]:
        #         print(res_c[0][xc])
        #         unnorm_save(context[0, :, 0, :, :].cpu(),224,str(time.time()))
        #        # transforms.ToPILImage()(context[0, :, 0, :, :][0].cpu()).save('./context.png')
        if self.context:
            #return res_f + res_c,res_c,res_f
            return res_f + res_c
        #  return context_score_fusion.div(context.size(2))
        else:
            return res_f


if __name__ == '__main__':
    model = Baseline(pretrain=False, context=False).cuda()
    # checkpoint = torch.load('../../pretrained_model/ResNet_pretrained.pth.tar')
    # load_state_dict = {k.replace('module.backbone.', ''): v for k, v in checkpoint['state_dict'].items() if
    #                    k.replace('module.backbone.', '') in model.backbone.state_dict()}
    # print(list(load_state_dict.items())[0])
    # model.backbone.load_state_dict(load_state_dict)
    # print(list(model.backbone.state_dict().items())[0])
    x = torch.rand(size=(16, 3, 3, 112, 112)).cuda()
    context = torch.rand(size=(8, 3, 8, 224, 224)).cuda()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print(model(context).size())
