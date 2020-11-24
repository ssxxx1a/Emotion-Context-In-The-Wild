'''
this file is use to create a Unify-train for all model
'''
import os
import timeit
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from model.Video_Model.reset3d import generate_model
from model.Video_Model.caen import CAEN
# from dataloaders.videodataset import VideoDataset
from dataloaders.Unify_Dataloader import Unify_Dataloader
from torch.utils.data import DataLoader
from utils.config import Model_Config, my_collate_fn, Ranger
from model.Video_Model.fan import resnet18_AT
from utils.FAN_load_materials import LoadParameter
from torch import optim
from utils.focal_loss import FocalLoss
from matplotlib import pyplot as plt
from model.Additional_model.model_irse import IR_50
from pycm import *
import model.For_test as For_test
from model.Video_Model.cnn_lstm import CNN_LSTM
from model.Video_Model.i3d_model import i3_res18_nl
from model.Video_Model.baseline import Baseline
import random
from pycm import *


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # consistent results on the cpu and gpu
    torch.backends.cudnn.benchmark = False

setup_seed(123)

def _worker_init_fn(worker_id: int) -> None:
    # Modulo 2**32 because np.random.seed() only accepts values up to 2**32 - 1
    initial_seed = torch.initial_seed() % 2 ** 32
    worker_seed = initial_seed + worker_id
    np.random.seed(worker_seed)

# Use GPU if available else revert to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device being used:", device)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
'''
just set this setting , it will be work
'''
setting = {'dataset': 'ours', 'model': 'baseline'}
# get config file
config = Model_Config()
common_config = Model_Config.get_common_config(setting['dataset'])
useTest = common_config['use_test']  # See evolution of the test set when training
nTestInterval = common_config['nTestInterval']  # Run on test set every nTestInterval epochs
snapshot = common_config['snapshot']  # Store a model every snapshot epochs
nEpochs = common_config['epoch']  # Number of epochs for training
save_dir = common_config['save_res_path']  # save path of confusion matrix and so on
dataset = common_config['dataset_name']  # used dataset name
num_classes = common_config['classes']  # num of classes
optim_name = common_config['optim']  # used optim name
bs = common_config['batch_size']  # batch size
Is_Context = common_config['is_context']
merge_frame_num = int(common_config['merge_frame_num'])  # how many frame to fusion
'''
 model param choose.
'''
model_config = config.get_model_config(setting['model'])
lr = model_config['lr']  # Learning rate
weight_decay = model_config['weight_decay']
modelName = model_config['model']  # Options: C3D or R2Plus1D or R3D
Ispretrain = model_config['Is_pretrain']

saveName = modelName
"""
Image_level model, just use the pretrain from offical weights

as following is Video-level pretrained model.
c3d : this pretrained is trained in sport-1M by ori C3D
r3d : this pretrained is trained in kinetics by r3d-50
fan : this pretrained is from debing Meng
"""
MODEL_PRETRAINED_PATH = {
    #'c3d': 'pretrained_model/c3d.pickle',
    'r3d': 'pretrained_model/r3d50_K_200ep.pth',
    'fan': 'pretrained_model/Resnet18_FER+_pytorch.pth.tar',
    'caen': 'pretrained_model/backbone_ir50_ms1m_epoch120.pth',
    'for_test': 'pretrained_model/backbone_ir50_ms1m_epoch120.pth',
    'baseline': 'pretrained_model/backbone_ir50_ms1m_epoch120.pth',
    'cnn_lstm': ''
}

"""
check path
"""
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
"""
remove the old log
"""
os.system('rm -rf ./log/*')
os.system('rm -rf ./Result/Confusion_matrix/*')


def Get_model(model_name, pretrain_model_path, pretrain=True):
    """
    :param model_name: the model name
    :param pretrain_model_path:  the path of pretained model
    :param pretrain: Is use Pretrain
    :return: a model (with pretained or not )
    note:
        this funtion is contain some code about diff lr for diff layer. if need,should be reference.
    """
    Model = {'fan': resnet18_AT(num_pair=merge_frame_num), 'r3d': generate_model(50),
             'caen': CAEN(merge_frame_num=merge_frame_num), 'for_test': i3_res18_nl(7),
             # 'for_test': For_test.ResNet_AT(),
             'baseline': Baseline(pretrain=pretrain, context=Is_Context), 'cnn_lstm': CNN_LSTM()}
    model = Model[model_name]
    if pretrain:
        if modelName == 'fan':  # at_type='self_relation-attention'
            # _structure = resnet18_AT(num_pair=3)
            _parameterDir = pretrain_model_path
            model = LoadParameter(model, _parameterDir)

        elif modelName == 'c3d':
            model.my_load_pretrained_weights(pretrain_model_path)
            # train_params = [{'params': get_1x_lr_params(model), 'lr': lr},
            #                 {'params': get_10x_lr_params(model), 'lr': lr * 10}]
            # elif modelName == 'R2Plus1D':
            #     model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
            #     train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
            #                     {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
            # elif modelName == 'R3D':
            #     model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
            #     train_params = model.parameters()
        elif modelName == 'r3d':
            model = model
        elif modelName == 'caen':
            checkpoint = torch.load(pretrain_model_path)
            model_stact_dict = model.backbone_face.state_dict()
            model_stact_dict.update(checkpoint)
        elif modelName == 'for_test':
            checkpoint = torch.load(pretrain_model_path)
            model_stact_dict = model.backbone.state_dict()
            model_stact_dict.update(checkpoint)
        elif modelName == 'baseline':
           # checkpoint = torch.load('./fer/fer+_best.pth.tar')
            checkpoint = torch.load('./pretrained_model/ResNet_pretrained.pth.tar')
            load_state_dict = {k.replace('module.backbone.', ''): v for k, v in checkpoint['state_dict'].items() if
                               k.replace('module.backbone.', '') in model.backbone.state_dict()}
            model.backbone.load_state_dict(load_state_dict)
        print('load weights finished')

    return model

def diff_lr(modelName, model):
    if modelName == 'caen':
        ig_params_face = list(map(id, model.backbone_face.parameters()))
        ig_params_img = list(map(id, model.backbone_img.parameters()))
        transforms_params = filter(
            lambda p: (id(p) not in ig_params_face) and (id(p) not in ig_params_img),
            model.parameters())
        params_list = [{'params': transforms_params, 'lr': 1e-4}]
        params_list.append({'params': model.backbone_face.parameters(), 'lr': 5e-4})
        params_list.append({'params': model.backbone_img.parameters(), 'lr': 5e-4})
        return params_list
    elif modelName == '_baseline':
        ig_params = list(map(id, model.backbone.parameters()))
        transforms_params = filter(lambda p: (id(p) not in ig_params), model.parameters())
        params_list = [{'params': transforms_params, 'lr': lr}, {'params': model.backbone.parameters(), 'lr': 1e-5}]
        return params_list
    else:
        params_list = [{'params': model.parameters(), 'lr': lr}]
        return params_list

def Trainer(dataset=dataset, save_dir=save_dir, lr=lr,
            num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    file = open(os.path.join(save_dir, 'log.txt'), 'w')
    print('---------------------settings of model as following----------------------------')
    for (name, set) in model_config.items():
        print(name + ' : ' + str(set))
    for (name, set) in common_config.items():
        print(name + ' : ' + str(set))
    print('optim : ' + optim_name)
    print('---------------------settings end----------------------------')
    model = Get_model(modelName, pretrain_model_path=MODEL_PRETRAINED_PATH[modelName], pretrain=Ispretrain)
    '''
    if use differ lr in differ layer ,
    you can reply the annotation in Get_model
    '''
    # train_params = model.parameters()
    # optimizer = config.get_optim_config(train_params, lr, weight_decay, optim_name)
    params_list = diff_lr(modelName, model)
    optimizer = Ranger(params_list, lr=lr, betas=(.95, 0.999), weight_decay=weight_decay)
    #  optimizer = optim.SGD(params_list, lr=1e-3, momentum=0.9, weight_decay=1e-4)
    #  scheduler=optim.lr_scheduler.ExponentialLR(optimizer, 0.95, -1)
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    # criterion = FocalLoss(class_num=7)
    #  scheduler = CosineAnnealingLR(optimizer, T_max=32, eta_min=0, last_epoch=-1)
    #   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9)
    #   optimizer = optim.SGD(params_list, lr=lr, momentum=0.9, weight_decay=5e-4)
    #   scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,
    #                                         gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    print("Training in {} Dataset".format(dataset))
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # model.to(device)
    if torch.cuda.is_available():
        model = model.cuda()
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model = nn.DataParallel(model)
        criterion.cuda()

    log_dir = os.path.join(save_dir)
    writer = SummaryWriter(log_dir=log_dir)
    print('Training model on {} dataset...'.format(dataset))

    # train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16, model_input_type=modelName),
    #                               batch_size=8,
    #                               shuffle=True, num_workers=8)
    val_dataloader = None
    # val_dataloader = DataLoader(
    #     VideoDataset(dataset='caer', split='validation', clip_len=16, model_input_type=modelName), batch_size=8,
    #     num_workers=8, drop_last=True, pin_memory=True)
    # test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16, model_input_type=modelName),
    #                              batch_size=8,
    #                              num_workers=8)

    train_dataloader = DataLoader(
        Unify_Dataloader(dataset_name=dataset, model_input_type=modelName, split='train', clip_len=merge_frame_num,
                         IsMark=True, Is_Context=Is_Context),
        batch_size=bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        collate_fn=my_collate_fn,
    )
    test_dataloader = DataLoader(
        Unify_Dataloader(dataset_name=dataset, split='test', model_input_type=modelName, clip_len=merge_frame_num,
                         IsMark=True, Is_Context=Is_Context),
        batch_size=bs,
        num_workers=8,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        collate_fn=my_collate_fn
    )
    # trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    # trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    trainval_loaders = {'train': train_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train']}
    test_size = len(test_dataloader.dataset)
    # my_smooth={'0': 0.88, '1': 0.95, '2': 0.96, '3': 0.79, '4': 0.65, '5': 0.89, '6': 0.88}
    best_acc_train = 0
    best_acc_test = 0
    confusion_pred = []
    confusion_label = []
    test_confusion_pred = []
    test_confusion_label = []
    save_acc = 0
    for epoch in range(1, num_epochs + 1):
        # each epoch has a training and validation step
        torch.cuda.empty_cache()
        for phase in ['train']:  # for phase in ['train','val]:
            start_time = timeit.default_timer()
            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0
            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                model.train()
            else:
                model.eval()
            for i, batch in enumerate(tqdm(trainval_loaders[phase])):
                face = batch['face']
                if Is_Context:
                    context = batch['context']
                labels = batch['label']
                # move inputs and labels to the device the training is taking place on
                face = face.to(device)
                labels = labels.to(device)
                if Is_Context:
                    context = context.to(device)
                # img = img.cuda(non_blocking=True)
                # labels = labels.cuda(non_blocking=True)
                if phase == 'train':
                    if Is_Context:
                        outputs = model(face, context)
                    else:
                        outputs = model(face)
                else:
                    with torch.no_grad():
                        if Is_Context:
                            outputs = model(face, context)
                        else:
                            outputs = model(face)

                probs = nn.Softmax(dim=1)(outputs)
                # the size of output is [bs , 7]
                preds = torch.max(probs, 1)[1]
                # print(preds.numpy())
                for xxx in range(len(preds)):
                    confusion_pred.append(int(preds[xxx].cpu()))
                    confusion_label.append(int(labels[xxx].cpu()))

                # preds is the index of maxnum of output
                # print(outputs)
                # print(torch.max(outputs, 1))

                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                #       scheduler.step(loss)
                # for name, parms in model.module.pred_fc2.named_parameters():
                #     # print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                #     #       ' -->grad_value:', parms.grad)
                #     if 'weight' in str(name):
                #         print('-->name:', name, ' -->grad_value:', parms[:10])

                running_loss += loss.item() * face.size(0)
                running_corrects += torch.sum(preds == labels.data)

            #  print('\ntemp/label:{}/{}'.format(preds[0], labels[0]))
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('data/train/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train/train_acc_epoch', epoch_acc, epoch)
            else:
                writer.add_scalar('data/val/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val/val_acc_epoch', epoch_acc, epoch)
            if best_acc_train < epoch_acc:
                best_acc_train = epoch_acc

            print("[{}] Epoch: {}/{} Loss: {} Acc: {} Best_Acc={}".format(phase, epoch, nEpochs, epoch_loss, epoch_acc,
                                                                          best_acc_train))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            file.write(
                "\n[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch, nEpochs, epoch_loss, epoch_acc))

        # if epoch % save_epoch == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'opt_dict': optimizer.state_dict(),
        #     }, os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar'))
        #     print("Save model at {}\n".format(
        #         os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar')))
        cm = ConfusionMatrix(confusion_label, confusion_pred)
        cm.save_html('Result/Confusion_matrix/train_' + str(epoch).zfill(3))
        confusion_pred = []
        confusion_label = []
        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for i, batch in enumerate(tqdm(test_dataloader)):
                face = batch['face']
                labels = batch['label']
                face = face.to(device)
                labels = labels.to(device)
                if Is_Context:
                    context = batch['context']
                    context = context.to(device)
                with torch.no_grad():
                    if Is_Context:
                        outputs = model(face, context)
                    else:
                        outputs = model(face)
                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                for xxx in range(len(preds)):
                    test_confusion_pred.append(int(preds[xxx].cpu()))
                    test_confusion_label.append(int(labels[xxx].cpu()))
                loss = criterion(outputs, labels)
                running_loss += loss.item() * face.size(0)
                running_corrects += torch.sum(preds == labels.data)
            #    print('\ntest: temp/label:{}/{}'.format(preds[0], labels[0]))
            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            writer.add_scalar('data/test/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test/test_acc_epoch', epoch_acc, epoch)
            if best_acc_test < epoch_acc:
                best_acc_test = epoch_acc
                _epoch = epoch
            print("[test] Epoch: {}/{} Loss: {} Acc: {} Best_acc:{} epoch_in={}".format(epoch, nEpochs, epoch_loss,
                                                                                        epoch_acc,
                                                                                        best_acc_test, _epoch))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")
            file.write("\n[test] Epoch: {}/{} Loss: {} Acc: {}\n".format(epoch, nEpochs, epoch_loss, epoch_acc))
        if save_acc < best_acc_test:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, saveName + '_best.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, saveName + '_epoch-' + str(epoch) + '.pth.tar')))
            save_acc = best_acc_test
            cm = ConfusionMatrix(test_confusion_label, test_confusion_pred)
            cm.save_html('Result/Confusion_matrix/best_test_' + str(epoch).zfill(3))
        test_confusion_pred = []
        test_confusion_label = []
    file.write('\nBest_train_acc:{} Best_test_acc:{} '.format(best_acc_train, best_acc_test))
    writer.close()
    file.close()


if __name__ == "__main__":
    Trainer(dataset=dataset, save_dir=save_dir, lr=lr, num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest,
            test_interval=nTestInterval)
    # Get_model(modelName, pretrain_model_path=MODEL_PRETRAINED_PATH[modelName], pretrain=Ispretrain)
