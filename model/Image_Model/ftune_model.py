import torch
import torch.nn as nn
import torchvision
from model.Additional_model.model_irse import IR_50
from model.Image_Model.resnet2 import ResNet18
import torch.nn.functional as F
from dataloaders.load_ferplus import Fer_loader, DataLoader


class Ftune_Model(nn.Module):
    def __init__(self):
        super(Ftune_Model, self).__init__()
        self.backbone = ResNet18()
        #  print(self.backbone)
        # self.backbone=nn.Sequential(
        #     *list(torchvision.models.resnet18(pretrained=False).children())[:-1]
        # )
        self.dropput = nn.Dropout(0.5)
        self.fc = nn.Linear(512, 8)

    def forward(self, x):
        res = self.backbone(x)
        res = res.view(-1, 512)
        res = self.dropput(res)
        res = self.fc(res)
        return F.softmax(res, dim=1)

if __name__ == '__main__':

    model = Ftune_Model()
    x = torch.rand(size=(8, 3, 48, 48))
    print(model(x))
    #  net=Ftune_Model()
    #  checkpoint = torch.load('../../pretrained_model/Resnet18.pth.tar')
    #  pretrained=checkpoint['net']
    #  #print(list(pretrained.items())[0])
    #  state_dict={k:v for k,v in pretrained.items() if k.replace('module.','') in net.backbone.state_dict()}
    #  state_dict={k.replace('module.',''):v for k,v in state_dict.items()}
    #  # print(list(state_dict.items())[0])
    #  # print(list(net.backbone.state_dict().items())[0])
    #  net.backbone.load_state_dict(state_dict)
    # # print(list(net.backbone.state_dict().items())[0])
