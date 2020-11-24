import torchvision
from model.Additional_model.model_irse import IR_50
from model.Additional_model.non_local_embedded_gaussian import NONLocalBlock2D
from model.Additional_model.Conv import *
from model.Additional_model.Action_transformer import Semi_Transformer


class Attention(nn.Module):
    def __init__(self, face_size):
        super(Attention, self).__init__()
        self.Conv_img = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            # nn.InstanceNorm2d(1),
            nn.ReLU(),
        )
        self.Conv_face = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            # nn.InstanceNorm2d(1),
            nn.ReLU(),
        )
        # self.Conv_img = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        # self.Conv_face = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=2)
        f_s = face_size * face_size
        self.q = nn.Linear(f_s, f_s)
        self.v = nn.Linear(f_s, f_s)
        self.k = nn.Linear(f_s, f_s)

    def forward(self, img, face):
        _img = img
        _face = face

        # _img = self.Conv_img(img)
        # _face = self.Conv_face(face)
        q, k, v = _face, _img, _img
        b, c, h, w = q.size()
        q = q.view(b, c, -1)
        k = k.view(b, c, -1)
        v = v.view(b, c, -1)
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        # second attention
        attention = torch.matmul(q.permute(0, 2, 1), k)
        attention = torch.matmul(F.softmax(attention, dim=1), v.permute(0, 2, 1)).permute(0, 2, 1)  # softmax
        attention = attention.view(b, c, h, w)
        #
        # attention = torch.matmul(q, k.permute(0, 2, 1))
        # attention = torch.matmul(self.softmax(attention), v)
        # attention = attention.view(b, c, h, w)
        #  print(attention.size())
        return attention


class CAEN(nn.Module):
    def __init__(self, merge_frame_num=16, nhead=8, d_model=512, classes=7, fm_size_face=7, fm_size_img=7,
                 at_type='self_relation-attention', is_context=True):
        super(CAEN, self).__init__()
        self.is_context = is_context
        self.merge_frame_num = merge_frame_num
        self.at_type = at_type
        #  self.backbone_face = IR_50([112,112])
        self.bn = nn.BatchNorm2d(512)
        self.backbone_face = nn.Sequential(
            *list(torchvision.models.resnet18(pretrained=True).children())[:-2],
        )
        # for param in self.backbone_face.parameters():
        #     param.requires_grad=False
        if self.is_context:
            self.context_bkb = nn.Sequential(
                *list(torchvision.models.resnet18(pretrained=True).children())[:-3],
                # nn.Conv2d(1024,512,4,2,1),
                # nn.BatchNorm2d(512),
                # nn.LeakyReLU()
            )
        self.vatn = Semi_Transformer(num_classes=7, seq_len=self.merge_frame_num)
        # self.Face_As_kenrels = custom_Conv_Net()
        #    self.non_local = NONLocalBlock2D(512)
        # self.as_some_size = nn.Conv2d(512, 512, 4, 2, 1)
        #  self.transformer = torch.nn.Transformer(d_model=d_model, nhead=nhead, dropout=0.5)
        # self.attention_inference_module = Attention(fm_size_face)
        self.fc = nn.Linear(512, classes)
        self.alpha = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.beta = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self.pred_fc1 = nn.Sequential(
            nn.Linear(512, classes),
            nn.Softmax(1)
        )
        self.pred_fc2 = nn.Sequential(
            nn.Linear(1024, classes),
            nn.Softmax(1)
        )

    def fusion(self, _context, _face, method):
        if not self.is_context or method == None:
            # print(_face.size())
            _face = self.bn(_face)
            _face = F.leaky_relu(_face)
            return F.adaptive_avg_pool2d(_face, 1)
        else:
            if method == 'nonlocal':
                '''
                则都是特征图
                '''
                context_fusion = self.non_local(_context, _face)
                _f = _face + context_fusion
                _f = F.adaptive_avg_pool2d(_f, 1)
                return _f
            elif method == 'transformer':
                bs = _context.size(0)
                _i = _context.view(bs, 512, -1).permute(2, 0, 1)
                _f = F.adaptive_avg_pool2d(_face, 1)
                _f = _f.view(bs, 512, -1).permute(2, 0, 1)
                context_fusion = self.transformer(_i, _f)  # fm_face.size(0),bs,512 =>[1,bs,512]
                _res = _f + context_fusion
                del (_i)
                del (_f)
                return _res
            elif method == 'F-transformer':
                bs, c, w, h = _context.size()
                _f = _face.view(bs, -1)
                _embed_f = self.theta_f(_f)
                _patch = []
                _f_s = _face.size(-1)
                _len = w - _f_s + 1
                # get patch, the len is 14-7+1
                for i in range(_len):
                    for j in range(_len):
                        _patch.append(_context[:, :, i:i + _f_s, j:j + _f_s])
                for i in range(len(_patch)):
                    _input_i = _patch[i].contiguous().view(bs, 512, -1)
                    _input_i = _input_i.permute(2, 0, 1)
                    _input_f = _face.view(bs, 512, -1)
                    _input_f = _input_f.permute(2, 0, 1)
                    _res = self.transformer(_input_i, _input_f)
                    _res = _res.permute(1, 2, 0)
                    _res = _res.view(bs, c, _f_s, _f_s)
            elif method == 'cat':
                context_fusion = torch.cat([_context, _face], dim=1)  # cat in c
                _f = self.fusion_cat(context_fusion)
                return _f
            elif method == 'attention':
                attention = self.attention_inference_module(_context, _face)
                _f = _face * attention + _face
                # _f = self.theta(_f)
                _f = F.adaptive_avg_pool2d(_f, 1)
                #  print(_f.size())
                # _f=_f.view(_f.size(0),-1)
                # _f=self.gamma(_f)
                return _f
            elif method == 'face_as_conv':
                bs, c, w, h = _face.size()
                _f = self.Face_As_kenrels(_context, _face)
                _f = F.adaptive_max_pool2d(_f, 7)
                _f = _face * _f + _face
                _f = F.adaptive_avg_pool2d(_f, 1)
                return _f
            #  _f = context_fusion.view(-1, 512)

    def forward(self, face, context=False):
        vs = []
        alphas = []
        bs, c, t, h, w = face.size()
        _vatn_data = face.permute(0, 2, 1, 3, 4)
        res_of_vatn = self.vatn(_vatn_data.clone())
        for i in range(t):
            _face = face[:, :, i, :, :]
            _face = self.backbone_face(_face)
            if self.is_context:
                _context = context[:, :, i, :, :]
                _context = self.context_bkb(_context)
            else:
                _context = None
            _f = self.fusion(_context, _face, method='None')  # (1,1,512)
            f = _f.view(-1, 512)
            #  print(f.size())
            fin_f = f
            vs.append(fin_f)
            alphas.append(self.alpha(F.dropout(fin_f, 0.5)))

        vs_stack = torch.stack(vs, dim=2)
        alphas_stack = torch.stack(alphas, dim=2)
        if self.at_type == 'self-attention':
            vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
            vm1 = self.dropout(vm1)
            pred_score = self.pred_fc1(vm1)
            pred_score = res_of_vatn[0] + pred_score
        elif self.at_type == 'self_relation-attention':
            vm1 = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
            betas = []
            for i in range(len(vs)):
                vs[i] = torch.cat([vs[i], vm1], dim=1)
                betas.append(self.beta(F.dropout(vs[i], p=0.5)))
            cascadeVs_stack = torch.stack(vs, dim=2)
            betas_stack = torch.stack(betas, dim=2)
            output = cascadeVs_stack.mul(betas_stack * alphas_stack).sum(2).div((betas_stack * alphas_stack).sum(2))
            output = F.dropout(output, p=0.6)
            pred_score = self.pred_fc2(output)
            pred_score = res_of_vatn[0] + pred_score
        else:
            raise NotImplementedError
        return pred_score


if __name__ == '__main__':
    net = CAEN()
    img = torch.rand(size=(16, 3, 16, 224, 224))
    face = torch.rand(size=(16, 3, 16, 112, 112))
    # image=Image.open('../../data/test_demo/00002.jpg').convert('RGB')
    # TTT=torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((224,224)),
    #     torchvision.transforms.ToTensor()
    # ])
    # ttt = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((112, 112)),
    #     torchvision.transforms.ToTensor()
    # ])
    # img=TTT(image)
    # face=ttt(image)
    # img=img.unsqueeze(dim=0).unsqueeze(dim=0)
    # img=img.permute(0,2,1,3,4)
    # face = face.unsqueeze(dim=0).unsqueeze(dim=0)
    # face = face.permute(0, 2, 1, 3, 4)
    # print(img.size())

    #
    #  print(net(img, face))
    #  print(net(img,face))
    #  net = nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])
    print(net(img, face).size())
    #
    # net= nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-2])
    # img=torch.rand(size=(10, 3, 224, 224))
    # print(net(img).size())
    # print(net( torch.rand(size=(10, 3, 224, 224))).size())
    # print(net)
    # _img = torch.rand(size=(10, 512, 14, 14))
    # _face = torch.rand(size=(10, 512, 7, 7))
    # A = Attention()
    # print(A(_img, _face))
