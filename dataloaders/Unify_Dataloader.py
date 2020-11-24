import os
import sys
import glob
import csv
import operator

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from torch.utils.data import Dataset, DataLoader
from utils.config import Dataset_Config, Model_Config
from sklearn.model_selection import train_test_split
from dataloaders.randomerase import RandomErasing
import torch
from dataloaders.deal_with_ECW import *
from torchvision import transforms
from PIL import Image

"""
this file is used to load video data:
such as UCF ...
"""


class Unify_Dataloader(Dataset):
    def __init__(self, dataset_name, model_input_type='caer', split='train', process=False, clip_len=16, IsMark=True,
                 Is_Context=True):
        super(Unify_Dataloader, self).__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.is_context = Is_Context
        self.model_input_type = model_input_type
        if self.dataset_name not in ['caer', 'ucf', 'ours']:
            raise NotImplementedError
        if not self.is_context:
            self.stream = 1
        else:
            self.stream = 2
        # special strem =0

        self.Dataser_Config = Dataset_Config()
        self.IsMark = IsMark
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112
        self.clip_len = clip_len
        self.Is_Expand_DATA = Model_Config.get_common_config(dataset_name)['expand_data']
        if self.split == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                #   RandomErasing(probability=0.5)
            ])
            self.threshold = Model_Config.get_common_config(dataset_name)['train_threshold']
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            self.threshold = Model_Config.get_common_config(dataset_name)['test_threshold']
        if process:
            # self.preprocess_ucf(self.dataset_name)
            self.preprocess(self.dataset_name)
        if not self.is_context:
            self.video_datas, self.labels = self.Init_Data(self.dataset_name, self.split)
        else:
            self.video_img, self.video_face, self.labels = self.Init_Data_caen()

    def Init_Data_caen(self):
        _, ECW_path = Dataset_Config.Get_Path('ours')

        face, face_label = self.Init_Data('ours', self.split, path=os.path.join(ECW_path, 'face'))

        if self.IsMark:
            img, img_label = self.Init_Data('ours', self.split, path=os.path.join(ECW_path, 'iwf'))
        else:
            img, img_label = self.Init_Data('ours', self.split, path=os.path.join(ECW_path, 'img'))

        if len(face_label) == len(img_label):
            return img, face, img_label
        else:
            print('the len of face and img label is not equal')
            raise NotImplementedError

    def preprocess(self, dataset_name):
        if dataset_name == 'ours':
            source_path, saved_path = Dataset_Config.Get_Path(dataset_name)
            temp_path = saved_path.replace(saved_path.split('/')[-1], 'Res')
            if not os.path.exists(source_path):
                print("your path of dataset is error")
                raise NotImplementedError
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
            if not os.path.exists(saved_path):
                os.mkdir(saved_path)
            setup_seed(123)
            generate_dir_res_from_videos(temp_path)
            generate_emotion_label(source_path)
            Generate_Data(source_path, temp_path)
            split_train_test(saved_path, temp_path, radio=0.7)
            os.system('rm -rf ' + temp_path)

    # for pre-process function.............
    def preprocess_ucf(self, dataset_name):
        """
        :param dataset_name:  the dataset you want to process
        :return:
        """
        source_path, saved_path = Dataset_Config.Get_Path(dataset_name)
        if not os.path.exists(source_path):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
            os.mkdir(os.path.join(saved_path, 'train'))
            os.mkdir(os.path.join(saved_path, 'val'))
            os.mkdir(os.path.join(saved_path, 'test'))
        for category_name in os.listdir(source_path):
            category_path = os.path.join(source_path, category_name)
            video_files = [name for name in os.listdir(category_path)]
            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(saved_path, 'train', category_name)
            val_dir = os.path.join(saved_path, 'val', category_name)
            test_dir = os.path.join(saved_path, 'test', category_name)
            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
            for video in tqdm(train):
                self.process_video(video=video, source_dir=os.path.join(source_path, category_name),
                                   save_dir=os.path.join(saved_path, 'train', category_name))
            for video in tqdm(val):
                self.process_video(video=video, source_dir=os.path.join(source_path, category_name),
                                   save_dir=os.path.join(saved_path, 'val', category_name))
            for video in tqdm(test):
                self.process_video(video=video, source_dir=os.path.join(source_path, category_name),
                                   save_dir=os.path.join(saved_path, 'test', category_name))
        print('finished process')

    def process_video(self, video, source_dir, save_dir, fps=10):
        video_name = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_name)):
            os.mkdir(os.path.join(save_dir, video_name))
        self.ffmpeg_run(video_name=os.path.join(source_dir, video), out_dir=os.path.join(save_dir, video_name), fps=fps,
                        resize=True)

    def process_bad_data(self, dataset_name, split, data_split):
        source_path, saved_path = Dataset_Config.Get_Path(dataset_name)
        print('process bad data  in ', saved_path)
        for k in data_split:
            if dataset_name == 'ours':
                saved_path = os.path.join(saved_path, k)

            folder = os.path.join(saved_path, split)
            for category in sorted(os.listdir(folder)):
                for fname in sorted(os.listdir(os.path.join(folder, category))):
                    fname_path = os.path.join(folder, category, fname)
                    if len(os.listdir(fname_path)) < 16:
                        print('remove:', fname_path)
                        cmd = 'rm -rf ' + fname_path
                        os.system(cmd)

    def ffmpeg_run(self, video_name, out_dir, fps=10, resize=True):

        if resize:
            size = str(self.resize_width) + 'x' + str(self.resize_height)
            com = "ffmpeg -i " + video_name + " -f image2 -vf fps=fps=10 -s " + str(size) + " " + \
                  os.path.join(out_dir, "out%3d.png")
        else:
            com = "ffmpeg -i " + video_name + " -f image2 -vf fps=fps=" + str(fps) + " " + \
                  os.path.join(out_dir, "out%3d.png")
        os.system(com)

    # for data-loader  function.............
    def Init_Data(self, dataset_name, split, path=None):
        """
        this function is used to get the list of processed video data
        :return:
        """
        if not path:
            source_path, saved_path = Dataset_Config.Get_Path(dataset_name)
            saved_path = os.path.join(saved_path, 'face')
        else:
            saved_path = path

        print('load data in ', saved_path)
        folder = os.path.join(saved_path, split)
        fnames, labels = [], []
        for category in sorted(os.listdir(folder)):
            count = 0
            list_of_emotion_file = sorted(os.listdir(os.path.join(folder, category)))
            for fname in list_of_emotion_file:

                fname_path = os.path.join(folder, category, fname)
                #   if len(os.listdir(fname_path))>self.clip_len:
                fnames.append(fname_path)
                labels.append(category)
                count += 1
                if count == self.threshold and self.Is_Expand_DATA and self.split != 'test':
                    break

            if count < self.threshold and self.Is_Expand_DATA and self.split != 'test':
                while count < self.threshold:
                    _i = random.randint(0, len(list_of_emotion_file) - 1)
                    fnames.append(os.path.join(folder, category, list_of_emotion_file[_i]))
                    labels.append(category)
                    count += 1

        assert len(labels) == len(fnames)
        print('Number of {} videos: {:d}'.format(split, len(fnames)))
        # Prepare a mapping between the label names (strings) and indices (ints)
        label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        label_array = np.array([label2index[label] for label in labels], dtype=int)

        return fnames, label_array

    def load_frames_with_context_by_simple(self, img_dir, face_dir, simple_len, crop_size):

        img_frames = sorted([os.path.join(img_dir, x) for x in os.listdir(img_dir)])
        face_frames = sorted([os.path.join(face_dir, x) for x in os.listdir(face_dir)])
        if len(img_frames) < len(face_frames):
            face_frames = face_frames[:len(img_frames)]
        else:
            img_frames = face_frames[:len(face_frames)]

        buffer_img = torch.empty(simple_len, 3, crop_size * 2, crop_size * 2)
        buffer_face = torch.empty(simple_len, 3, crop_size, crop_size)
        # 不使用随机crop
        clip_n = len(img_frames) // simple_len
        if clip_n < 1:
            # print('failed in ', frames[0])
            print(img_frames)
            return None
        for i in range(simple_len):
            if self.split == 'train':
                clip_index = np.random.randint(i * clip_n, (i + 1) * clip_n)
            else:
                clip_index = i * clip_n
            img = Image.open(img_frames[clip_index])
            face = Image.open(face_frames[clip_index])
            img = img.resize((224, 224))
            face = face.resize((112, 112))
            img = self.transform(img)
            buffer_img[i] = img
            face = self.transform(face)
            buffer_face[i] = face
        buffer_face = buffer_face.permute(1, 0, 2, 3)
        buffer_img = buffer_img.permute(1, 0, 2, 3)
        return buffer_img, buffer_face

    def load_frames_by_simple(self, file_dir, crop_size, simple_len):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])

        buffer = torch.empty(simple_len, 3, crop_size, crop_size)
        # size_w = np.random.randint(self.resize_height - crop_size)  # 128裁剪112
        # size_h = np.random.randint(self.resize_height - crop_size)
        clip_n = len(frames) // simple_len

        if clip_n < 1:
            # print('failed in ', frames[0])
            print('clip_n:', clip_n)
            print('file_dir:', file_dir)
            return None
        for i in range(simple_len):
            if self.split == 'train':
                clip_index = np.random.randint(i * clip_n, (i + 1) * clip_n)
            else:
                clip_index = clip_n * i
            #  print(frames[clip_index])
            img = Image.open(frames[clip_index])
            img = img.resize((112, 112))
            if img:
                #   img = img.crop((size_w, size_h, size_w + crop_size, size_h + crop_size))
                img = self.transform(img)
                buffer[i] = img
        buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_frames(self, file_dir, clip_len, crop_size):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        if len(frames) >= clip_len:
            if len(frames) == clip_len:
                load_frames = frames
            else:
                start = np.random.randint(len(frames) - clip_len)
                load_frames = frames[start:start + clip_len]
        else:
            copy_count = int(clip_len - len(frames))
            copy_frame = frames[-1]
            for i in range(copy_count):
                frames.append(copy_frame)
            load_frames = frames
        buffer = torch.empty(clip_len, 3, crop_size, crop_size)
        size_w = np.random.randint(self.resize_width - crop_size)
        size_h = np.random.randint(self.resize_height - crop_size)
        for i, frame_name in enumerate(load_frames):
            img = Image.open(frame_name)
            img = img.crop((size_w, size_h, size_w + crop_size, size_h + crop_size))
            img = self.transform(img)
            buffer[i] = img

        buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def __getitem__(self, index):
        labels = np.array(self.labels[index])
        if self.stream == 0:
            buffer = self.load_frames(self.video_datas[index], clip_len=self.clip_len, crop_size=self.crop_size)
            return {'face': buffer, 'label': torch.from_numpy(labels)}
        elif self.stream == 1:
            buffer = self.load_frames_by_simple(self.video_datas[index], crop_size=self.crop_size,
                                                simple_len=self.clip_len)
            return {'face': buffer, 'label': torch.from_numpy(labels)}
        elif self.stream == 2:
            buffer_img, buffer_face = self.load_frames_with_context_by_simple(self.video_img[index],
                                                                              self.video_face[index],
                                                                              simple_len=self.clip_len,
                                                                              crop_size=self.crop_size)
            return {'context': buffer_img, 'face': buffer_face, 'label': torch.from_numpy(labels)}

    def __len__(self):
        return len(self.labels)


unloader = transforms.ToPILImage()


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


if __name__ == '__main__':
    # train_dataloader = DataLoader(
    #     Unify_Dataloader(dataset_name='ours', model_input_type='fan', split='train', clip_len=3,process=True,
    #                      IsMark=True),
    #     batch_size=8,
    #     shuffle=True,
    #     pin_memory=True,
    # )
    data = Unify_Dataloader(dataset_name='ours', model_input_type='fan', process=False, split='train', clip_len=16,
                            IsMark=True)
    for i in ['train', 'test']:
        data.process_bad_data('ours', i, ['img', 'iwf', 'face'])
    #
    # for x in train_dataloader:
    #     for xx in range(x['img'].size(2)):
    #         img = x['img'][0][:, xx, :, :]
    #         img = tensor_to_PIL(img)
    #         img.save(str(xx) + '_.png')
    #         print(x['label'][0])
    #     break

    # data = Unify_Dataloader(dataset_name='ours', model_input_type='fan', process=False, split='train', clip_len=16,
    #                         IsMark=True)
    # for i in ['train', 'test']:
    #     data.process_bad_data('ours', i, ['img', 'iwf', 'face'])
    # for i, batch in enumerate(train_dataloader):
    #     img = batch['img']
    #     face = batch['face']
    #     img.save('x.png')
    #     break
    # x.ffmpeg_run('123', '456', 10)
