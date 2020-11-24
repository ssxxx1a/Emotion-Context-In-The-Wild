import torch.nn.functional as F
import torch
import torch.nn as nn
from model.Image_Model.basemodel import BaseModel


class Encoder(nn.Module):
    def __init__(self, num_kernels, kernel_size=3, bn=True, max_pool=True, maxpool_kernel_size=2, isContext=True):
        super().__init__()
        padding = (kernel_size - 1) // 2
        n = len(num_kernels) - 1
        self.convs = nn.ModuleList(
            [nn.Conv2d(num_kernels[i], num_kernels[i + 1], kernel_size, padding=padding) for i in range(n)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(num_kernels[i + 1]) for i in range(n)]) if bn else None
        self.max_pool = nn.MaxPool2d(maxpool_kernel_size) if max_pool else None
        self.isContext = isContext
        if self.isContext:
            self.adaptive = nn.AdaptiveMaxPool2d(3)

    def forward(self, x):
        n = len(self.convs)
        for i in range(n):
            x = self.convs[i](x)
            if self.bn is not None:
                x = self.bn[i](x)
            x = F.relu(x)
            if self.max_pool is not None:  # check if i < n
                x = self.max_pool(x)
        if self.isContext:
            return self.adaptive(x)
        else:
            return x


class TwoStreamNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        num_kernels = [3, 32, 64, 128, 256, 256]
        self.face_encoding_module = Encoder(num_kernels, isContext=False)
        self.context_encoding_module = Encoder(num_kernels)
        self.attention_inference_module = Encoder([256, 128, 1], max_pool=False)

    def forward(self, face, context):
        face = self.face_encoding_module(face)

        context = self.context_encoding_module(context)
        print(context.size())
        attention = self.attention_inference_module(context)
        print(context.size())
        N, C, H, W = attention.shape
        attention = F.softmax(attention.reshape(N, C, -1), dim=2).reshape(N, C, H, W)
        context = context * attention

        return face, context

class FusionNetwork(nn.Module):
    def __init__(self, num_class=7):
        super().__init__()
        self.face_1 = nn.Linear(256 * 9, 128)
        self.face_2 = nn.Linear(128, 1)

        self.context_1 = nn.Linear(256 * 9, 128)
        self.context_2 = nn.Linear(128, 1)

        self.fc1 = nn.Linear(512 * 9, 128)
        self.fc2 = nn.Linear(128, num_class)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, face, context):
        # face = F.avg_pool2d(face, face.shape[2]).reshape(face.shape[0], -1)
        # context = F.avg_pool2d(context, context.shape[2]).reshape(context.shape[0], -1)
        face = face.reshape(face.shape[0], -1)
        context = context.reshape(context.shape[0], -1)

        lambda_f = F.relu(self.face_1(face))
        lambda_c = F.relu(self.context_1(context))

        lambda_f = self.face_2(lambda_f)
        lambda_c = self.context_2(lambda_c)

        weights = torch.cat([lambda_f, lambda_c], dim=-1)
        weights = F.softmax(weights, dim=1)

        face = face * weights[:, 0].unsqueeze(dim=-1)
        context = context * weights[:, 1].unsqueeze(dim=-1)

        features = torch.cat([face, context], dim=-1)
        features = self.dropout1(features)

        features = F.relu(self.fc1(features))

        features = self.dropout2(features)

        return self.fc2(features)


class CAERSNet(BaseModel):
    def __init__(self):
        super().__init__()
        self.two_stream_net = TwoStreamNetwork()
        self.fusion_net = FusionNetwork()

    def forward(self, context=None, face=None):
        face, context = self.two_stream_net(face, context)

        return F.softmax(self.fusion_net(face, context), dim=1)


if __name__ == '__main__':
    net=TwoStreamNetwork()
    a=torch.rand(size=(10,3,112,112))
    b = torch.rand(size=(10, 3, 224, 224))
    net(a,b)

#  vaild_distribution = {'0': 0, '1': 12312, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
#  emtion_label = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
# # for i in range(len(vaild_distribution)):
#  keys=list(vaild_distribution.keys())
#  emotion_keys=list(emtion_label.keys())
#  for i in range(len(vaild_distribution)):
#      vaild_distribution.update({emotion_keys[i]:vaild_distribution.pop(keys[i])})
#  print(vaild_distribution)
