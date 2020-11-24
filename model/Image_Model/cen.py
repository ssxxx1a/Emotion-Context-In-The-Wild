import torch.nn as nn
import torchvision
import torch.nn.functional as F
from model.Image_Model.basemodel import BaseModel
from dataloaders.dataLoader import *


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CEN(BaseModel):

    def __init__(self, nhead=8, d_model=512):
        super(CEN, self).__init__()
        self.backbone_face = nn.Sequential(
            *list(ResNet(fm_size=3).children())[:-1]
            # net18(pretrained=True),
            # nn.ReLU(),
            # nn.AdaptiveMaxPool2d(1)
        )
        self.backbone_img = nn.Sequential(
            *list(ResNet(fm_size=3).children())[:-1]
            # resnet18(pretrained=True),
            # nn.ReLU(),
            # nn.AdaptiveMaxPool2d(7)
        )
        self.transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead, dropout=0.5)
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(in_planes=512)
        self.attention_fc = nn.Linear(49, 7)
        self.fc = nn.Linear(512, 7)

        '''
        non-local block
        '''
        self.g = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.W=nn.Sequential(
            nn.Conv2d(512,512,3,2,0),
            nn.BatchNorm2d(512)
        )
        self.fc=nn.Linear(512,7)
    # self.softmax = nn.Softmax(dim=2)

    # for i in self.backbone_face.parameters():
    #     i.requires_grad = False
    # for i in self.backbone_img.parameters():
    #     i.requires_grad = False
    # for i in self.transformer.modules():
    #     if isinstance(i, (nn.Conv2d, nn.Linear)):
    #         nn.init.kaiming_uniform_(i.weight, mode='fan_in', nonlinearity='relu')
    # for i in self.fc.modules():
    #     if isinstance(i, (nn.Conv2d, nn.Linear)):
    #         nn.init.kaiming_uniform_(i.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, img, face, transformer=False, cbam=True):
        fm_face = self.backbone_face(face)  # size: [bs,512,1,1]
        fm_img = self.backbone_img(img)  # size: [bs,512,7,7]
        bs = fm_img.size(0)
        dim = fm_img.size(1)
        # print(fm_face.size(), fm_img.size())
        fm_face_embed = fm_face.view(-1, bs, dim)  # size:[sz*sz,bs,512]
        fm_img_embed = fm_img.view(-1, bs, dim)  # size:[sz*sz,bs,512]
        # print(fm_face_embed.size(), fm_img_embed.size())
        '''
        transformer
        '''
        if transformer:
            res = F.relu(self.transformer(fm_img_embed, fm_face_embed))
            # size: [1,bs,512]
            # res = torch.cat([self.transformer(fm_img_embed, fm_face_embed),fm_face_embed],dim=-1)
            res += fm_face_embed
            res = self.fc(res)
            res = (F.softmax(res, dim=2)).squeeze(0)
            return res
        else:
            res = fm_face_embed + fm_img_embed
        '''
        fm_face=[bs,512,1,1]
        fm_img=[bs,512,7,7]
        we can compute this as  [bs*1*1 ,512]*[512, bs*7*7] use the matmul. and use a softmax -> this as attention
        [bs,512,1,1]
        [bs,1,7,7]
        '''
        if cbam:
            g_face=self.g(fm_face).reshape(bs,512,-1)
            g_face=g_face.permute(0,2,1)
            fm_face = self.theta(fm_face).reshape(bs,512,-1)
            fm_face=fm_face.permute(0,2,1)
            fm_img=self.phi(fm_img).reshape(bs,512,-1)
            res=torch.matmul(F.softmax(torch.matmul(fm_face,fm_img),dim=-1),g_face).view(bs,512,3,3)
            res=self.W(res)
            res=res.view(bs,-1)
            fin_res=self.fc(res)
            return F.softmax(fin_res,dim=1)

            # fm_face=fm_face.reshape(bs,-1,512)#[bs,9,512]
            # fm_img=fm_img.reshape(bs,512,-1)#[bs,512,9]
            # attention=F.softmax(F.relu(torch.matmul(fm_face,fm_img)),dim=2)
            # #res=torch.matmul(fm_face,fm_img)
            # print(attention.size())
            # #attention=attention.reshape(bs,-1,512)
            # res=fm_face+attention
            # res=res.reshape(bs,-1,3,3)
            # print(fm_face.size())
            # print(fm_img.size())
            # print(res)
        #     print(fm_face)

        # fm_face = self.ca(fm_face) * fm_face
        # fm_face = fm_face.view(bs, 512, -1)
        # fm_face = fm_face.permute(0, 2, 1)
        # # print(fm_face.size())
        # fm_img = (self.ca(fm_img) * fm_img).view(bs, 512, -1)
        # # print(fm_img.size())
        # res = self.up_c(torch.matmul(fm_face, fm_img)) + fm_face
        # res = self.fc(res).squeeze()
        # # print(res.size())
        return F.softmax(res, dim=1)
    #
    #  # res=res.squeeze(0)
    #
    #  res = self.fc(res)
    # # res = self.softmax(res).squeeze(0)
    #  # cv_format = np.transpose(res.detach(), (1, 0, 2)).view(bs, -1)
    #  return F.softmax(res,dim=2).squeeze(0)


class ResNet(BaseModel):
    def __init__(self, fm_size=1):
        super(ResNet, self).__init__()
        pretrain_model = torchvision.models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(
            *list(pretrain_model.children())[:-2],
            nn.AdaptiveAvgPool2d(fm_size)
        )
        nf = pretrain_model.fc.in_features
        # *list(torchvision.models.resnet152(pretrained=True).children())[:-2],
        # for i in self.backbone.parameters():
        #     i.requires_grad = False
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(nf, 7)
        )
        # for i in self.fc.modules():
        #     if isinstance(i,(nn.Conv2d, nn.Linear)):
        #         nn.init.kaiming_normal_(i.weight, mode='fan_in')

    def forward(self, face):
        bs = face.size(0)
        x = self.backbone(face)
        x = x.view(bs, -1)
        res = self.fc(x)
        # bs = face.size(0)
        # x = self.backbone(face).view(bs, -1)
        # y = self.softmax(self.fc(x))
        return F.softmax(res, dim=1)


if __name__ == '__main__':
    root = '/opt/data/private/dbmeng/Data/Emotion/Caer/Caer-S/train'
    txt_path = '../../data/label_file/mini_face_info.txt'
    # print(d[1]['face'].size())

    face = torch.rand(size=(10, 3, 96, 96))
    img = torch.rand(size=(10, 3, 224, 224))
    net = CEN()
    print(net(img, face).size())

    '''
    以下为加载预训练模型
    '''
    # backbone_face_path = '/opt/data/private/pycharm_map/Context-emotion/saved_model/FER_resnet_face_50.pth'
    # backbone_img_path = '/opt/data/private/pycharm_map/Context-emotion/saved_model/FER_resnet_img_50.pth'
    # cen = CEN()
    # param_face_dict = torch.load(backbone_face_path)
    # face_dict = cen.backbone_face.state_dict()
    # pretrained_face_dict = {k.strip('module.').replace('backbone', '0'): v for k, v in param_face_dict.items() if
    #                         k.strip('module.').replace('backbone', '0') in face_dict}
    # face_dict.update(pretrained_face_dict)
    # cen.backbone_face.load_state_dict(face_dict)
    # # load backbone_img pretrain param
    # param_img_dict = torch.load(backbone_img_path)
    # img_dict = cen.backbone_img.state_dict()
    # pretrained_img_dict = {k.strip('module.').replace('backbone', '0'): v for k, v in param_img_dict.items() if
    #                        k.strip('module.').replace('backbone', '0') in img_dict}
    # face_dict.update(pretrained_img_dict)
    # cen.backbone_img.load_state_dict(img_dict)

    # load backbone_face pretrain param
    # param_face_dict = torch.load(backbone_face_path)
    # face_dict = cen.backbone_face.state_dict()
    # pretrained_face_dict = {k.strip('module.'): v for k, v in param_face_dict.items() if
    #                         k.strip('module.') in face_dict}
    # face_dict.update(pretrained_face_dict)
    # cen.backbone_face.load_state_dict(face_dict)
    # # load backbone_img pretrain param
    # param_img_dict = torch.load(backbone_img_path)
    # img_dict = cen.backbone_img.state_dict()
    # pretrained_img_dict = {k.strip('module.'): v for k, v in param_img_dict.items() if
    #                        k.strip('module.') in img_dict}
    # face_dict.update(pretrained_img_dict)
    # cen.backbone_face.load_state_dict(img_dict)

# print(model_dict)


# pretrained_dict = {k.strip('module.'): v for k, v in param_dict.items() if k.strip('module.') in model_dict}
# print(pretrained_dict)
# print(model_dict.keys())

# check_point=torch.load(backbone_face_path)
# model_dict=ResNet().state_dict()
# pretrained_dict = {k: v for k, v in check_point.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_state_dict(model_dict)


#  DD=dlib.get_frontal_face_detector()
#  img=cv2.imread('0024.png')
#  ssss=DD(img,1)
#  face=ssss[0]
#  res = img[face.top():face.bottom(), face.left():face.right()]
#  res = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGRA2RGB))
#
#  ttt=transforms.Compose([
#      transforms.Resize((96,96),Image.BICUBIC),
#      transforms.ToTensor()
#  ])
#  res=ttt(res).unsqueeze(0)
#  R=ResNet()
#  print(R(res))

# N = CEN()
# print(N.backbone_face)
# res=N(img, face)
# print(res)
# print(torch.sum(res[0]))
# # print(N(img, face,).size())
# R = ResNet()
# for name,param in R.named_parameters():
#     print('-->name:', name, '-->grad_requirs:', param.requires_grad, \
#           ' -->grad_value:', param.grad)
#  res = R(img)
# print(res)
# print(torch.sum(res[0]))
