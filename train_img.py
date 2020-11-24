'''

this train is use to train ECW-S data, for the video task is useless.
'''

import argparse
from tensorboardX import SummaryWriter
from evaluate import *
from model.Image_Model.cen import *
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys

'''
1.
model: caer
lr:5e-3 
optim: SGD weight_decay 1e-5
input  96 96, 
with scheduler
without initial param
-->not work
2.
model: caer
lr:5e-5
optim: Adam weight_decay 1e-5
input  224 96
without scheduler 
without initial param
res:[Epoch 10/50] [Batch 676/677] [loss: 1.603386] [example ~ argamx_index:3 label:3][batch_acc:0.578125 total_acc:0.589526]
3.
model: caer
lr:5e-5
optim: SGD weight_decay 1e-5
input  224 96
without scheduler 
without initial param
--> not work
4.
model: caer
lr:5e-3
optim: SGD weight_decay 1e-5
input  224 96
with  scheduler 10 decey each 4 epoch
without initial param
--> not work
5.
model: caer
lr:5e-5
optim: Adam weight_decay 1e-5
input  224 96
without scheduler 
with initial param
--> work , better than 2. but small category 'angry' have bad res
res: . valid set: 35%

6.
model: caer
lr: 5e-5
optim: adam weight_decay 1e-5
input   224,96
without scheduler 
without initial param 
res:
[Epoch 50/50] [Batch 675/677] [loss: 1.414026] [example ~ argamx_index:4 label:2][batch_acc:0.750000 total_acc:0.795095]
[Epoch 50/50] [Batch 676/677] [loss: 1.316405] [example ~ argamx_index:1 label:1][batch_acc:0.843750 total_acc:0.795167]
for test ......

 epoch:49 ,train_acc:0.795167 , vaild_acc:0.341727 ,pred_distribution {'Angry': 0, 'Disgust': 111, 'Fear': 39, 'Happy': 111, 'Neutral': 231, 'Sad': 29, 'Surprise': 35}

-----------------------------------------------cen
1.
model: cen
lr:5e-5
optim: Adam weight_decay 1e-5
input  224 96
without scheduler 
with initial param
---> not work . pred always is one label

2.
model: cen
lr:1e-4
optim: Adam weight_decay 1e-5
input  224 96
without scheduler 
with initial param
---> not work . 

3.
---> not work . ---> not work . ---> not work . 
lr: from 1e-3 ~ 5e-7 all can't work
model: cen
lr:1e-6
optim: Adam weight_decay 1e-5
input  224 96
without scheduler 
without initial param

4.
model: cen
lr:3e-4
optim: Adam 
weight_decay: 1e-4
input: 224 96
scheduler: CosineAnnealingLR
initial_param : kaiming_uni

---------------------------------------resnet
1.
 model: res, 
 lr :1e-4, 
 optim, Adam weight_decay 1e-5
 input :96, 
 scheduler:False 
 initial_param:False
 -->not work
 
 2.
model: res
 lr :1e-4, 
 optim, Adam weight_decay 0 
 input :224, 
 scheduler:False 
 initial_param:False
'''
parser = argparse.ArgumentParser()
parser.add_argument("--train_type", type=str, default='cen', help="choose the model")
parser.add_argument("--nepoch", type=int, default=50, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--face_size", type=int, default=96, help="face size")
parser.add_argument("--img_size", type=int, default=224, help="img size")
parser.add_argument("--lr", type=float, default=1e-3, help="optim lr")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="optim weight_decay")
parser.add_argument("--If_scheduler", type=bool, default=True, help="if use scheduler")
parser.add_argument("--If_finetune", type=bool, default=True, help="if use finetune")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--train_img_path", type=str, default='/opt/data/private/dbmeng/Data/Emotion/Caer/Caer-S/train',
                    help="the path of data for train")
parser.add_argument("--vaild_img_path", type=str, default='/opt/data/private/dbmeng/Data/Emotion/Caer/Caer-S/test',
                    help="the path of data for train")
parser.add_argument("--save_model_path", type=str, default='./saved_model',
                    help="the path of saved model")
parser.add_argument("--checkpoint_interval", type=int, default=10,
                    help="the path of saved model")
opt = parser.parse_args()
print(opt)

# fix random seeds for reproducibility
# SEED = 123
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# np.random.seed(SEED)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = True if torch.cuda.is_available() else False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# loss function
# optimizer_G = torch.optim.Adam(Net.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# model = CEN()
# root = '/Users/arthur/Documents/data/MinCaer/test'
train_data = DataLoader(
    EmotionDataset(opt.train_img_path, txt_path='data/label_file/train.txt', img_size=opt.img_size,
                   face_size=opt.face_size, IsMark=True),
    batch_size=opt.batch_size,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    num_workers=16,
    collate_fn=my_collate_fn
)
vaild_data = DataLoader(
    EmotionDataset(opt.vaild_img_path, txt_path='data/label_file/test.txt', img_size=opt.img_size, face_size=opt.face_size),
    batch_size=opt.batch_size,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    num_workers=16,
    collate_fn=my_collate_fn
)
LOG_DIR = {'caer': 'caer', 'cen': 'cen', 'resnet': 'baseline', 'resnet_face': 'resnet_face', 'resnet_img': 'resnet_img'}
MODEL = {'caer': CAERSNet(), 'cen': CEN(), 'resnet': ResNet()}
backbone_face_path = '/opt/data/private/pycharm_map/Context-emotion/saved_model/FER_resnet_face_50.pth'
backbone_img_path = '/opt/data/private/pycharm_map/Context-emotion/saved_model/FER_resnet_img_50.pth'


def train(train_type, res_dir='resnet'):
    os.system('rm -rf ./log/' + LOG_DIR[res_dir] + '/*')
    writer = SummaryWriter(log_dir='./log/' + LOG_DIR[res_dir] + '/')
    log_file = open('./log_' + LOG_DIR[res_dir] + '.txt', mode='w')
    model = MODEL[train_type]
    print(model)
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()
    if opt.If_finetune:
        if train_type == 'resnet':
            # .module是遍历全部子层
            ignored_params = list(map(id, model.fc.parameters()))
            base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
            params_list = [{'params': base_params, 'lr': opt.lr / 10}]
            params_list.append({'params': model.fc.parameters(), 'lr': opt.lr})
        elif train_type == 'cen':
            # for i in model.backbone_face.parameters():
            #     i.requires_grad=False
            # for i in model.backbone_img.parameters():
            #     i.requires_grad=False
            # finetune1 = list(map(id, model.module.backbone_face.parameters()))
            # finetune2 = list(map(id, model.module.backbone_img.parameters()))
            # base_params = filter(lambda p: (id(p) not in finetune1) and (id(p) not in finetune2), model.parameters())
            # params_list = [{'params': base_params, 'lr': opt.lr}]
            # # finetune 的lr 小于 train的
            # params_list.append({'params': model.module.backbone_face.parameters(), 'lr': opt.lr / 10})
            # params_list.append({'params': model.module.backbone_img.parameters(), 'lr': opt.lr / 10})
            param_face_dict = torch.load(backbone_face_path)
            face_dict = model.backbone_face.state_dict()
            pretrained_face_dict = {k.strip('module.').replace('backbone', '0'): v for k, v in param_face_dict.items()
                                    if
                                    k.strip('module.').replace('backbone', '0') in face_dict}
            face_dict.update(pretrained_face_dict)
            model.backbone_face.load_state_dict(face_dict)
            # load backbone_img pretrain param
            param_img_dict = torch.load(backbone_img_path)
            img_dict = model.backbone_img.state_dict()
            pretrained_img_dict = {k.strip('module.').replace('backbone', '0'): v for k, v in param_img_dict.items() if
                                   k.strip('module.').replace('backbone', '0') in img_dict}
            img_dict.update(pretrained_img_dict)
            model.backbone_img.load_state_dict(img_dict)
            # 将backbone设置为参数不可得
            # for i in model.backbone_face.parameters():
            #     i.requires_grad = False
            # for i in model.backbone_img.parameters():
            #     i.requires_grad = False
            ignored_params_face = list(map(id, model.backbone_face.parameters()))
            ignored_params_img = list(map(id, model.backbone_img.parameters()))
            transforms_params = filter(
                lambda p: (id(p) not in ignored_params_face) and (id(p) not in ignored_params_img),
                model.parameters())
            params_list = [{'params': transforms_params, 'lr': opt.lr}]
            params_list.append({'params': model.backbone_face.parameters(), 'lr': opt.lr / 10})
            params_list.append({'params': model.backbone_img.parameters(), 'lr': opt.lr / 10})

        optim = torch.optim.Adam(params_list, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay,
                                 amsgrad=True)
    else:
        # optim = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        optim = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay,
                                 amsgrad=True)
        # optim = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=1e-4)
    if opt.If_scheduler:
        # 学习率衰减
        scheduler = CosineAnnealingLR(optim, T_max=32, eta_min=0, last_epoch=-1)
        # scheduler = StepLR(optim, gamma=0.1, step_size=4)
    if cuda:
        # 设置为多卡训练
        torch.backends.cudnn.benchmark = True
        # 先cuda 再DataParallel封装
        model = model.cuda()
        model = nn.DataParallel(model)
        criterion = criterion.cuda()
    nepoch = opt.nepoch
    for epoch in range(nepoch):
        model.train()
        count = 0
        total_count = 0
        for i, batch in enumerate(train_data):
            input_face = batch['face']
            input_img = batch['img']
            input_label = batch['label']
            label_temp = input_label
            # t = transforms.ToPILImage()
            # input_label = F.one_hot(input_label, 7)
            # input_label = input_label.type(Tensor)
            # put data into GPU
            if torch.cuda.is_available():
                input_face = input_face.cuda(non_blocking=True)
                input_label = input_label.cuda(non_blocking=True)
                input_img = input_img.cuda(non_blocking=True)
            if train_type != 'resnet':
                res = model(input_img, input_face)
            else:
                if res_dir == 'resnet_face':
                    res = model(input_face)
                elif res_dir == 'resnet_img':
                    res = model(input_img)
            loss = criterion(res, input_label)
            optim.zero_grad()
            # loss.requires_grad = True
            loss.backward()
            # for name, parms in model.module.backbone_img.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
            #           ' -->grad_value:', parms.grad)
            optim.step()
            if opt.If_scheduler:
                scheduler.step()
            # size of res is [8 x 6] ->[bs x category_num]
            temp_count = 0
            for xx in range(len(res)):
                if int(res[xx].argmax().item()) == int(label_temp[xx].item()):
                    count += 1
                    temp_count += 1
            total_count += len(label_temp)
            batch_acc = temp_count / len(label_temp)
            total_acc = count / total_count
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [example ~ argamx_index:%d label:%d][batch_acc:%f total_acc:%f]\n"
                % (
                    epoch + 1,
                    nepoch,
                    i + 1,
                    len(train_data),
                    loss.item(),
                    int(res[0].argmax().item()),
                    int(label_temp[0]),
                    batch_acc,
                    total_acc
                )
            )
            if (i + 1) % 10 == 0:
                step = epoch * len(train_data) + (i + 1)
                writer.add_scalar('train/loss', loss, step)
        print('for test ......')
        vaild_count = 0
        vaild_distribution = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
        emtion_label = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
        keys = list(vaild_distribution.keys())
        emotion_keys = list(emtion_label.keys())
        model.eval()

        for index, e in tqdm(enumerate(vaild_data)):
            vaild_face = e['face']
            vaild_label = e['label']
            vaild_img = e['img']
            if torch.cuda.is_available():
                vaild_face = vaild_face.cuda(non_blocking=True)
                vaild_label = vaild_label.cuda(non_blocking=True)
                vaild_img = vaild_img.cuda(non_blocking=True)

            if train_type != 'resnet':
                with torch.no_grad():
                    vaild_res = model(vaild_img, vaild_face)
            else:
                if res_dir == 'resnet_face':
                    with torch.no_grad():
                        vaild_res = model(vaild_face)
                elif res_dir == 'resnet_img':
                    with torch.no_grad():
                        vaild_res = model(vaild_img)
            vaild_loss = criterion(vaild_res, vaild_label)
            for xx in range(len(vaild_res)):
                if int(vaild_res[xx].argmax().item()) == int(vaild_label[xx].item()):
                    vaild_count += 1
                vaild_distribution[str(vaild_res[xx].argmax().item())] += 1
            if (index + 1) % 10 == 0:
                vaild_step = epoch * len(vaild_data) + (index + 1)
                writer.add_scalar('test/loss:', vaild_loss, vaild_step)

        vaild_acc = vaild_count / vaild_data.__len__() / opt.batch_size
        writer.add_scalar('train/acc', total_acc, epoch)
        writer.add_scalar('test/acc:', vaild_acc, epoch)

        # just for replace annotation
        for i in range(len(vaild_distribution)):
            vaild_distribution.update({emotion_keys[i]: vaild_distribution.pop(keys[i])})
        # print(vaild_distribution)

        log_file.write('epoch:%d ,train_acc:%f , vaild_acc:%f ,pred_distribution %s\n' % (
            epoch + 1, total_acc, vaild_acc, str(vaild_distribution)))
        print('\n epoch:%d ,train_acc:%f , vaild_acc:%f ,pred_distribution %s\n' % (
            epoch + 1, total_acc, vaild_acc, str(vaild_distribution)))
        if opt.checkpoint_interval != -1 and (epoch + 1) % opt.checkpoint_interval == 0:
            # Save model checkpoints
            # 似乎这样才能保存并行训练的模型
            torch.save(model.state_dict(), "%s/FER_%s_%d.pth" % (opt.save_model_path, LOG_DIR[res_dir], epoch + 1))
    writer.close()
    log_file.close()


if __name__ == '__main__':
    # print('train:  resnet face')
    # train(train_type='resnet', res_dir='resnet_face')
    # print('train:  resnet img')
    # train(train_type='resnet', res_dir='resnet_img')
    # *list(torchvision.models.resnet152(pretrained=True).children())[:-2],
    train(opt.train_type, res_dir=opt.train_type)
