import os
import torch.nn as nn
import torch
import glob
from ranger import Ranger
from torch.utils.data.dataloader import default_collate
import sys


def my_collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return torch.Tensor()
    return default_collate(batch)


class Dataset_Config(object):

    @staticmethod
    def Get_Path(dataset_name):
        DATASET_PATH = {
            'caer': '/opt/data/private/dbmeng/Data/Emotion/Caer/Caer',
            'ucf': '/opt/data/private/data/ucf/ori_data/',
            # 'ours': '/opt/data/private/data/Refine_ECW/ECW',
         #   'ours': '/opt/data/private/data/R-ecw'
            'ours': '/opt/data/private/data/FR_ecw'
        }
        SAVED_PATH = {
            'caer': '/opt/data/private/data/processCaer/',
            'ucf': '/opt/data/private/data/ucf/process_data',
            'ours': '/opt/data/private/data/FR_ecw_split'
            #'ours': '/opt/data/private/data/R_ecw_split'

            # 'ours': '/opt/data/private/data/Refine_ECW/ECW_split',
        }
        return DATASET_PATH[dataset_name], SAVED_PATH[dataset_name]

    @staticmethod
    def get_fer_path(split):
        if split == 'train':
            img_path = '/opt/data/private/project/fer+/FER2013Train'
            label_path = '/opt/data/private/project/FERPlus/data/FER2013Train/label.csv'
        elif split == 'val':
            img_path = '/opt/data/private/project/fer+/FER2013Valid'
            label_path = '/opt/data/private/project/FERPlus/data/FER2013Valid/label.csv'
        else:
            img_path = '/opt/data/private/project/fer+/FER2013Test'
            label_path = '/opt/data/private/project/FERPlus/data/FER2013Test/label.csv'
        return img_path, label_path


class Model_Config(object):
    @staticmethod
    def get_common_config(dataset):

        setting = {
            'dataset_name': dataset,
            'dataset_path': '',
            'save_model_path': './saved_model',
            'save_res_path': './log',
            'epoch': 100,
            'nTestInterval': 1,  # Run on test set every nTestInterval epochs
            'snapshot': 10,  # Store a model every snapshot epochs
            'classes': 7,
            'use_test': True,
            'optim': 'Ranger',
            'is_context': False,
            'merge_frame_num': 16,
            'train_threshold': 300,
            'test_threshold': 100,
            'expand_data': False,
            'batch_size': 8
        }
        return setting

    @staticmethod
    def get_optim_config(train_params, lr, weight_decay, optim_name):

        if optim_name == 'Ranger':
            return Ranger(train_params, lr=lr, betas=(.95, 0.999), weight_decay=weight_decay)
        elif optim_name == 'SGD':
            return torch.optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optim_name == 'Adam':
            return torch.optim.Adam(train_params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay, amsgrad=True)
        else:
            # 默认
            return Ranger(train_params, lr=lr, betas=(.95, 0.999), weight_decay=weight_decay)

    @staticmethod
    def get_pretrain_config(model_name):
        """
        Image_level model, just use the pretrain from offical weights

        as following is Video-level pretrained model.
        c3d : this pretrained is trained in sport-1M by ori C3D
        r3d : this pretrained is trained in kinetics by r3d-50
        fan : this pretrained is from debing Meng
        """
        MODEL_PRETRAINED_PATH = {
            # 'c3d': 'pretrained_model/c3d.pickle',
            'r3d': 'pretrained_model/r3d50_K_200ep.pth',
            'fan': 'pretrained_model/Resnet18_FER+_pytorch.pth.tar',
            'caen': 'pretrained_model/backbone_ir50_ms1m_epoch120.pth',
            'for_test': 'pretrained_model/backbone_ir50_ms1m_epoch120.pth',
            'baseline': 'pretrained_model/backbone_ir50_ms1m_epoch120.pth',
            'cnn_lstm': ''
        }
        if model_name not in MODEL_PRETRAINED_PATH.keys():
            return None
        else:
            return MODEL_PRETRAINED_PATH[model_name]

    @staticmethod
    def get_model_config(model_name):
        CAER = {
            'model': 'caer',
            'lr': 1e-5,
            'weight_decay': 1e-5,
            'Is_pretrain': True,
            'Is_scheduler': True
        }
        FAN = {
            'model': 'fan',
            'lr': 3e-4,  # 4e-6,
            'weight_decay': 1e-4,
            'Is_pretrain': True,
            'Is_scheduler': True
        }
        CAEN = {
            'model': 'caen',
            'lr': 3e-4,  # 4e-6,
            'weight_decay': 1e-4,
            'Is_pretrain': False,
            'Is_scheduler': True,
        }
        FOR_TEST = {
            'model': 'for_test',
            'lr': 3e-4,  # 4e-6,
            'weight_decay': 1e-4,
            'Is_pretrain': False,
            'Is_scheduler': True
        }
        BASELINE = {
            'model': 'baseline',
            'lr': 3e-4,  # 4e-6,
            'weight_decay': 1e-4,
            'Is_pretrain': False,
            'Is_scheduler': True
        }
        CNN_LSTM = {
            'model': 'cnn_lstm',
            'lr': 5e-5,  # 4e-6,
            'weight_decay': 1e-4,
            'Is_pretrain': False,
            'Is_scheduler': True
        }
        model = {'caer': CAER, 'fan': FAN, 'caen': CAEN, 'for_test': FOR_TEST, 'baseline': BASELINE,
                 'cnn_lstm': CNN_LSTM}
        return model[model_name]
