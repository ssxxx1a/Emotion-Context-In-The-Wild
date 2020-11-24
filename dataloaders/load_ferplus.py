import csv
import operator
from torch.utils.data import Dataset
import os
from config import Dataset_Config
from torchvision import transforms
from PIL import Image
import cv2
emtion_label = {
    '0': 'neutral', '1': 'happiness', '2': 'surprise', '3': 'sadness', '4': 'anger', '5': 'disgust', '6': 'fear',
    '7': 'contempt'}

class Fer_loader(Dataset):
    def __init__(self, split):
        super(Fer_loader, self).__init__()
        self.split=split
        if split=='train':
            self.t = transforms.Compose([
            #    transforms.Grayscale(),  # 使用ImageFolder默认扩展为三通道，重新变回去就行
                transforms.RandomHorizontalFlip(),  # 随机翻转
                transforms.ColorJitter(brightness=0.5, contrast=0.5),  # 随机调整亮度和对比度
                transforms.ToTensor()
            ])
        else:
            self.t = transforms.Compose([
        #        transforms.Grayscale(),
                transforms.ToTensor()
            ])

        f_path, l_path = Dataset_Config.get_fer_path(split)
        self.data, self.labels = self.load_ferplus(l_path, f_path)

    def mv_data_to_IR50(self):
        path=os.path.join('/opt/data/private/project/face.evoLVe.PyTorch/data',self.split)
        if not os.path.exists(path):
            os.mkdir(path)
        for k in emtion_label.keys():
            if not os.path.exists(os.path.join(path,emtion_label[k])):
                os.mkdir(os.path.join(path,emtion_label[k]))
        for i in range(self.__len__()):
            cmd='cp '+self.data[i]+' '+os.path.join(path,emtion_label[str(self.labels[i])])
            os.system(cmd)

    def load_ferplus(self, label_file_path, img_folder_path):
        img_list = []
        label_list = []
        with open(label_file_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                max_index, max_number = max(enumerate(row[2:]), key=operator.itemgetter(1))
                fname = row[0]
                label = max_index
                if label > 7:
                    continue
                img_list.append(os.path.join(img_folder_path, fname))
                label_list.append(label)
        return img_list, label_list

    def __getitem__(self, index):

        #img = Image.open(self.data[index])
        src=cv2.imread(self.data[index])
        img = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        i = self.t(img)
        label = self.labels[index]
        return {'img': i, 'label': label}

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    x = Fer_loader('test')
    # x.mv_data_to_IR50()
