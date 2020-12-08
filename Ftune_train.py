'''
use to train fer+ pretrain
'''
import argparse
import os
import torch
import torch.nn as nn
from  model.Image_Model.ftune_model import Ftune_Model
from torch.utils.data import DataLoader
from dataloaders.load_ferplus import Fer_loader
from tqdm import tqdm
from config import Ranger
parser = argparse.ArgumentParser(description='Net')
parser.add_argument('--nepoch', type=int, default=50, help='num of epoch')
parser.add_argument('--lr', type=int, default=1e-3, help='lr')
parser.add_argument('--weight_decay', type=int, default=0, help='weight_decay')
parser.add_argument('--bs', type=int, default=128, help='batch size')
opt = parser.parse_args()
device='cuda' if torch.cuda.is_available() else 'cpu'
def trainer():
    model = Ftune_Model()
   # checkpoint = torch.load('./pretrained_model/Resnet18.pth.tar')
  #  pretrained = checkpoint['net']
    # print(list(pretrained.items())[0])
  #  state_dict = {k: v for k, v in pretrained.items() if k.replace('module.', '') in model.backbone.state_dict()}
  #  state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # print(list(state_dict.items())[0])
    # print(list(net.backbone.state_dict().items())[0])
   # model.backbone.load_state_dict(state_dict)
    # print(list(net.backbone.state_dict().items())[0])

   # optim=torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=opt.weight_decay, amsgrad=True)
    optim= Ranger(model.parameters(),lr=opt.lr, betas=(.95, 0.999), weight_decay=opt.weight_decay)
  #  scheduler = lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
  #  scheduler = CosineAnnealingLR(optim, T_max=32, eta_min=0, last_epoch=-1)
    criterion=nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model = nn.DataParallel(model)
        criterion.cuda()

    print('===> Loading datasets')
    train_data = DataLoader(
        Fer_loader('train'),
        batch_size=opt.bs,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    test_data = DataLoader(
        Fer_loader('test'),
        shuffle=True,
        batch_size=opt.bs,
        pin_memory=True,
        drop_last=True,
        num_workers=8
    )
    file = open(os.path.join('./log', 'log.txt'), 'w')
    print('len of train',train_data.__len__())
    print('len of test', test_data.__len__())
    best_acc_train = 0
    best_acc_test = 0
    for epoch in range(0,opt.nepoch):
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        for i,batch in enumerate(tqdm(train_data)):
            img=batch['img'].to(device)
            label = batch['label'].to(device)
            optim.zero_grad()
            output=model(img)
            probs = nn.Softmax(dim=1)(output)
            # the size of output is [bs , 10]
            preds = torch.max(probs, 1)[1]
            loss = criterion(output, label)
            loss.backward()
            optim.step()
       #     scheduler.step()
            running_loss += loss.item() * img.size(0)
            running_corrects += torch.sum(preds == label.data)
        #    print('\ntemp/label:{}/{}'.format(preds[0], label[0]))
        epoch_loss = running_loss / (len(train_data)*opt.bs)
        epoch_acc = running_corrects.double() /(len(train_data)*opt.bs)
        if best_acc_train < epoch_acc:
            best_acc_train = epoch_acc

        print("[{}] Epoch: {}/{} Loss: {} Acc: {} Best_Acc={}".format('train', epoch, opt.nepoch, epoch_loss, epoch_acc,
                                                                          best_acc_train))
        file.write(
            "\n[{}] Epoch: {}/{} Loss: {} Acc: {}".format('train', epoch, opt.nepoch, epoch_loss, epoch_acc))
        # if (epoch+1)%1==0:
        #     torch.save({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'opt_dict': optim.state_dict(),
        #     }, os.path.join('./fer', 'fer+_' + '_epoch-' + str(epoch+1) + '.pth.tar'))
        running_loss = 0.0
        running_corrects = 0.0
        for i, batch in enumerate(tqdm(test_data)):
            model.eval()
            img = batch['img'].to(device)
            label = batch['label'].to(device)

            with torch.no_grad():
                output = model(img)
            loss = criterion(output, label)
            probs = nn.Softmax(dim=1)(output)
            # the size of output is [bs , 10]
            preds = torch.max(probs, 1)[1]
            running_loss += loss.item() * img.size(0)
            running_corrects += torch.sum(preds == label.data)
        epoch_loss = running_loss /(len(test_data)*opt.bs)
        epoch_acc = running_corrects.double() / (len(test_data)*opt.bs)
        if best_acc_test < epoch_acc:
            best_acc_test = epoch_acc
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt_dict': optim.state_dict(),
            }, os.path.join('./fer', 'fer+_' + 'best.pth.tar'))
        print("[{}] Epoch: {}/{} Loss: {} Acc: {} Best_Acc={}".format('test', epoch, opt.nepoch, epoch_loss, epoch_acc,
                                                                      best_acc_test))
        file.write("\n[test] Epoch: {}/{} Loss: {} Acc: {}\n".format(epoch, opt.nepoch, epoch_loss, epoch_acc))
    file.write('\nBest_train_acc:{} Best_test_acc:{} '.format(best_acc_train, best_acc_test))
    file.close()

if __name__ == '__main__':

    trainer()