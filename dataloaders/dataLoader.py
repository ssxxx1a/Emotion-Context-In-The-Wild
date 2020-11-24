from torch.utils.data import Dataset
from dataloaders.preprocess import Generate_data_txt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import torch
import dlib
import cv2
import os
from torch.utils.data.dataloader import default_collate

class EmotionDataset(Dataset):
    def __init__(self, imgFolder_path, txt_path='./face_info.txt', img_size=224, face_size=96, IsMark=False):
        # imgFolder_path is used to creaet txt_file if txx_file is not exist
        super(EmotionDataset, self).__init__()
        self.imgFolder_path = imgFolder_path
        self.txt_path = txt_path
        self.detector = dlib.get_frontal_face_detector()
        #self.detector = dlib.cnn_face_detection_model_v1("../data/mmod_human_face_detector.dat")
        self.ismarkface = IsMark
        self.sp = dlib.shape_predictor(
            '/opt/data/private/pycharm_map/Context-emotion/shape_predictor_68_face_landmarks.dat')
        self.img_size = img_size
        self.face_size = face_size
        self.img_transform, self.face_transform = self.get_img_transform()
        self.CommonImg = self.GetDataFromTxt()

    def load_data_all(self):
        '''
        将全部img处理好全部load进显存大概占据8个G。
        :return:
        '''
        return None

    def get_img_transform(self):
        '''
        get  transforms which to deal with context img and face
        :return:
        '''
        img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),

            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        face_transofrm = transforms.Compose([
            transforms.Resize((self.face_size, self.face_size), Image.BICUBIC),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return img_transform, face_transofrm

    def detect(self, gimg):
        """
        :param gimg:  这个是cv2.imread
        :param detector: dlib
        :return:
        """
        face = self.detector(gimg, 1)
        # 只读取第一张人脸（数据集里也就只有一张）
        if len(face) > 0:
            face = face[0]
            res = gimg[face.top():face.bottom(), face.left():face.right()]
            try:
                img = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGRA2RGB))
            except:
                print('\ndetect error!!!!!!')
                return None
            return img
        else:
            print('\ndetect error!!!!!!')
            return None

    def GetDataFromTxt(self):
        '''
        get formated data from txt.
        the content of txt is :
        [path:face_area:emtion]
        split with ':'

        :return:
        '''
        if not os.path.exists(self.txt_path):
            print('txt file is not exist,creating txt.......')
            Generate_data_txt(self.imgFolder_path, self.txt_path, is_face=False)
        Common_img = []
        emtion_label = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
        format = ['jpg', 'png', 'jepg']
        file = open(self.txt_path, 'r')
        print('load img info txt: ', self.txt_path)
        '''
        use the offical train.txt
        '''
        if self.txt_path.split('/')[-1] == 'train.txt' or self.txt_path.split('/')[-1] == 'test.txt':
            for i in tqdm(file.readlines()):
                temp = i.split(',')
                img_path = os.path.join(self.imgFolder_path, temp[0])
                if not os.path.exists(img_path):
                    continue
                # data format: Surprise/5928.png,6,268,47,366,145
                label = int(temp[1])
                face_area = [int(i) for i in temp[2:]]
                # 改为top bottom left right
                face_area = [face_area[1], face_area[3], face_area[0], face_area[2]]
                Common_img.append({'img': img_path, 'face': face_area, 'label': label})
        elif self.txt_path.split('/')[-1] == 'crop_face_train.txt' or \
                self.txt_path.split('/')[-1] == 'crop_face_test.txt':
            for i in tqdm(file.readlines()):
                img_path = i.split(':')[0]
                if img_path.split('.')[-1] in format:
                    emotion = i.split(':')[-1].rstrip('\n')
                    label = emtion_label[emotion]
                Common_img.append({'img': None, 'face': img_path, 'label': label})
        else:
            # for the txt of myself
            for i in tqdm(file.readlines()):
                img_path = i.split(':')[0]
                if img_path.split('.')[-1] in format:
                    face_area = i.split(':')[1].lstrip('[').rstrip(']').split(',')
                    face_area = [int(x) for x in face_area]
                    emtion = i.split(':')[2].rstrip('\n')
                    label = emtion_label[emtion]
                    # img = cv2.imread(img_path)
                    # face = img[face_area[0]:face_area[1], face_area[2]:face_area[3]]
                    Common_img.append({'img': img_path, 'face': face_area, 'label': label})
        print('data of ' + self.txt_path + ' distribution is :')
        self.__getDataDistribution__()
        return Common_img

    def __getitem__(self, index, ):
        '''
        if the data['img'] is None, it is mean that input info_txt is only face info.

        :param index:  index of data
        :param isMarkFace:  Does it need to be marked.
        :return:
        '''
        data = self.CommonImg[index]

        label = data['label']
        img_path = data['img']
        face_area = data['face']
        '''
         for new
         if the face area are beyond the border,we need to find that
        '''
        if img_path != None:
            try:
                img = cv2.imread(img_path)
                # crop the face from img by face area
                # face_area[0]: top ,face_area[1]: bottom ,face_area[2]: left ,face_area[3]: right
                face = img[face_area[0]: face_area[1], face_area[2]:face_area[3]]
                # use dlib to make face align
                area = dlib.rectangle(face_area[2], face_area[0], face_area[3], face_area[1])
                face = dlib.get_face_chip(img, self.sp(face, area), size=self.face_size)
                # transpose opencv to PIL.Image
                face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGRA2RGB))
                face = self.face_transform(face)
                if self.ismarkface:
                    img[face_area[0]: face_area[1], face_area[2]:face_area[3]] = 0
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
                img = self.img_transform(img)
                return {'img': img, 'face': face, 'label': label}
            except:
                return None
        else:
            # 这时候 ,face_area保存的是path,不再是人脸区域
            try:
                face = cv2.imread(face_area)
                face = self.face_transform(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGRA2RGB)))
                return {'face': face, 'label': label}
            except:
                return None

    # for i in range(len(face_area)):
    #     if face_area[i] <= 0:
    #         _area[i] = 1face
    # face = img[face_area[0]: face_area[1], face_area[2]:face_area[3]]
    # face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGRA2RGB))
    # face = self.face_transform(face)
    # if isMarkFace:
    #     img[face_area[0]: face_area[1], face_area[2]:face_area[3]] = 0
    # # print('face:', face_area)
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
    # img = self.img_transform(img)
    # return {'img': img, 'face': face, 'label': label}

    # try:
    #     top, bottom, left, right = face_area[0], face_area[1], face_area[2], face_area[3]
    #     face = img.crop((left, top, right, bottom))
    #     img = self.img_transform(img)
    #     if isMarkFace:
    #         img[top:bottom,left:right]=0
    #     face = self.face_transform(face)
    #     t=transforms.ToPILImage()
    #     t(img).save(index+'_img.png')
    #     return {'img': img, 'face': face, 'label': label}
    #
    # except:
    #     return None

    def __len__(self):
        return len(self.CommonImg)

    def __getDataDistribution__(self):
        file = open(self.txt_path)
        emtion_label = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}
        if self.txt_path.split('/')[-1] != 'train.txt' and self.txt_path.split('/')[-1] != 'test.txt':
            for i in file.readlines():
                e = i.split(':')[-1].rstrip('\n')
                emtion_label[e] += 1
            print(emtion_label)
        else:
            for i in file.readlines():
                e = i.split('/')[0]
                emtion_label[e] += 1
            print(emtion_label)


# class Evalue_dataloader(Dataset):
#     def __init__(self, txt_path, img_size=96, face_size=96):
#         super(Evalue_dataloader, self).__init__()
#         self.txt_path = txt_path
#         self.img_size = img_size
#         self.face_size = face_size
#         self.img_transform, self.face_transform = self.get_img_transform()
#         self.CommonSet = self.data_init()
#
#     def get_img_transform(self):
#         '''
#         get  transforms which to deal with context img and face
#         :return:
#         '''
#         img_transform = transforms.Compose([
#             transforms.Resize((self.img_size, self.img_size), Image.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])
#         face_transofrm = transforms.Compose([
#             transforms.Resize((self.face_size, self.face_size), Image.BICUBIC),
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ])
#         return img_transform, face_transofrm
#
#     def data_init(self):
#         if not os.path.exists(self.txt_path):
#             print('txt file is not exist,creating txt.......')
#             # '/opt/data/private/data/Caer/minCaer/Test' is path of  img folder
#             Generate_data_txt('/opt/data/private/data/Caer/minCaer/Test', self.txt_path)
#         Common_img = []
#         emtion_label = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
#         format = ['jpg', 'png', 'jepg']
#         file = open(self.txt_path, 'r')
#         print('\rload vaild data.....')
#         for i in tqdm(file.readlines()):
#             img_path = i.split(':')[0]
#             if img_path.split('.')[-1] in format:
#                 face_area = i.split(':')[1].lstrip('[').rstrip(']').split(',')
#                 face_area = [int(x) for x in face_area]
#                 emtion = i.split(':')[2].rstrip('\n')
#                 label = emtion_label[emtion]
#                 # img = cv2.imread(img_path)
#                 # face = img[face_area[0]:face_area[1], face_area[2]:face_area[3]]
#                 Common_img.append({'img': img_path, 'face': face_area, 'label': label})
#         return Common_img
#
#     def __getitem__(self, index):
#         data = self.CommonSet[index]
#         label = data['label']
#         img_path = data['img']
#         img = cv2.imread(img_path)
#         face_area = data['face']
#         for i in range(len(face_area)):
#             if face_area[i] <= 0:
#                 face_area[i] = 1
#         face = img[face_area[0]: face_area[1], face_area[2]:face_area[3]]
#         # print('face:', face_area)
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGB))
#         face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGRA2RGB))
#         img = self.img_transform(img)
#         face = self.face_transform(face)
#         return {'img': img, 'face': face, 'label': label}
#
#     def __len__(self):
#         return len(self.CommonSet)
#
#     def __getDataDistribution__(self):
#         file = open(self.txt_path)
#         emtion_label = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}
#         for i in file.readlines():
#             e = i.split(':')[-1].rstrip('\n')
#             emtion_label[e] += 1
#         print(emtion_label)


if __name__ == '__main__':
    root = '/opt/data/private/dbmeng/Data/Emotion/Caer/Caer-S/train'
    txt_path = '../data/label_file/train.txt'
    # print(d[1]['face'].size())
    d = EmotionDataset(root, txt_path)
    print(d.__getDataDistribution__())
    # print(d[0])
    # # print(img)
    # #  img.save('1.png')
    # data = DataLoader(
    #     EmotionDataset(root, txt_path),
    #     batch_size=8,
    #     pin_memory=True,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=16,
    #     collate_fn=my_collate_fn
    # )

    # print(d.__getDataDistribution__())
    #
    # # t=transforms.ToPILImage()
    # for i, batch in enumerate(data):
    #     print(batch)
    # print(img.size)
    # res=img.crop((0,120,100,200))
    # res.save('2.png')
    # img.save('1.png')
    # break

# img=cv2.imread('/opt/data/private/data/Caer/Caer-S/train/Fear/0688.png')
# print(img)
# model = CEN()
# count = 0
# vaild_data = DataLoader(
#     Evalue_dataloader('./face_test_info.txt'),
#     batch_size=8,
#     pin_memory=True,
#     shuffle=True,
#     drop_last=True,
#     num_workers=8
# )
# for index, e in enumerate(vaild_data):
#     img = e['img']
#     face = e['face']
#     label = e['label']
#     out = model(img, face)
#     # out=out.argmax().item()
#     for xx in range(len(out)):
#         if int(out[xx].argmax().item()) == int(label[xx].item()):
#             count += 1
#         print(out[xx].argmax().item(), int(label[xx].item()))
