import os
import sys
import dlib
import cv2
import math
from tqdm import tqdm
import random
import numpy as np

IGNORE_DATA = []


# IGNORE_DATA= ['S0002', 'S0038', 'S0039', 'S0040', 'S0054', 'S0063', 'S0074', 'S0081', 'S0085', 'S0087', 'S0099', 'S0102',
#           'S0117', 'S0151', 'S0223', 'S0224', 'S0225', 'S0226', 'S0239', 'S0241', 'S0242', 'S0243', 'S0245', 'S0259',
#           'S0266', 'S0292', 'S0294', 'S0303', 'S0306', 'S0320', 'S0332', 'S0335', 'S0384', 'S0386', 'S0399', 'S0400',
#           'S0404', 'S0425', 'S0436', 'S0439', 'S0482', 'S0499', 'S0532', 'S0539', 'S0543', 'S0550', 'S0563', 'S0564',
#           'S0567', 'S0572', 'S0579', 'S0582', 'S0583', 'S0585', 'S0590', 'S0594', 'S0595', 'S0599', 'S0600', 'S0628',
#           'S0629', 'S0630', 'S0659', 'S0660', 'S0682', 'S0684', 'S0686', 'S0687', 'S0692', 'S0717', 'S0719', 'S0720',
#           'S0744', 'S0756', 'S0781', 'S0782', 'S0784', 'S0785', 'S0787', 'S0832', 'S0835', 'S0841', 'S0845', 'S0848',
#           'S0862', 'S0864', 'S0868', 'S0870', 'S0878', 'S0919', 'S0930', 'S0931', 'S0932', 'S0951', 'S0989', 'S1024',
#           'S1035', 'S1055', 'S1056', 'S1081', 'S1083', 'S1100', 'S1102', 'S1134', 'S1135', 'S1136', 'S1150', 'S1170',
#           'S1171', 'S1174', 'S1192', 'S1195', 'S1197', 'S1200', 'S1210', 'S1245', 'S1246', 'S1256', 'S1258', 'S1259',
#           'S1262', 'S1280', 'S1281', 'S1283', 'S1342', 'S1366', 'S1373', 'S1374', 'S1378', 'S1379', 'S1383', 'S1392',
#           'S1400', 'S1405', 'S1406', 'S1438', 'S1448', 'S1452', 'S1458', 'S1463', 'S1464', 'S1473', 'S1482', 'S1506',
#           'S1507', 'S1509', 'S1510', 'S1511', 'S1521', 'S1522', 'S1523', 'S1531', 'S1552', 'S1561', 'S1563', 'S1564',
#           'S1567', 'S1609', 'S1611', 'S1632', 'S1633', 'S1634', 'S1636']
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_landmark_data_from_file(path):
    F = open(path, 'r+')
    data = F.readlines()
    F.close()
    return data


def get_coordinates_from_landmark(landmark, index1=None, index2=None, begin_index=0):
    # begin_index is the begin of array ,default is from zero.
    if index1 and index2:
        x_1 = landmark[index1 * 2 - 2 + begin_index]
        y_1 = landmark[index1 * 2 - 1 + begin_index]

        x_2 = landmark[index2 * 2 - 2 + begin_index]
        y_2 = landmark[index2 * 2 - 1 + begin_index]
        return (x_1, y_1), (x_2, y_2)
    elif index1:
        x_1 = landmark[index1 * 2 - 2 + begin_index]
        y_1 = landmark[index1 * 2 - 1 + begin_index]
        return (x_1, y_1)
    elif index2:
        x_2 = landmark[index2 * 2 - 2 + begin_index]
        y_2 = landmark[index2 * 2 - 1 + begin_index]
        return (x_2, y_2)


def computer_distance(point1, point2):
    x = (point1[0] - point2[0]) * (point1[0] - point2[0])
    y = (point1[1] - point2[1]) * (point1[1] - point2[1])
    return math.sqrt(x + y)


def center_to_point(center, width, height):
    left_top = (int(center[0] - width / 2), int(center[1] + height / 2))
    right_bottom = (int(center[0] + width / 2), int(center[1] - height / 2))

    return left_top, right_bottom


def get_lefttop_and_rightbottom(landmark, w, h):
    cor_1, cor_17 = get_coordinates_from_landmark(landmark, 1, 17)
    width = int(computer_distance(cor_1, cor_17) * w)
    cor_9, cor_28 = get_coordinates_from_landmark(landmark, 9, 28)
    height = int(computer_distance(cor_9, cor_28) * h) * 1.5
    #  print(width, height)
    center = get_coordinates_from_landmark(landmark, 31)
    center = int(center[0] * w), int(center[1] * h)
    # print(center)
    left_top, right_bottom = center_to_point(center, width, height)
    return left_top, right_bottom


def get_lefttop_and_rightbottom_with_border(landmark, w, h, border_list):
    cor_1, cor_17 = get_coordinates_from_landmark(landmark, 1, 17)
    width = int(computer_distance(cor_1, cor_17) * w)
    cor_9, cor_28 = get_coordinates_from_landmark(landmark, 9, 28)
    height = int(computer_distance(cor_9, cor_28) * h) * 1.5
    #  print(width, height)
    center = get_coordinates_from_landmark(landmark, 31)
    center = int(center[0] * w), int(center[1] * h)
    #   print(center)
    # [left, right, top, bottom]
    #
    left_top = (int(border_list[0] * w - width // 15), int(border_list[2] * h - height // 10))

    right_bottom = (int(border_list[1] * w + width // 15), int(border_list[3] * h))

    return left_top, right_bottom


def get_border_list(landmark_x, landmark_y):
    left = min(landmark_x)
    right = max(landmark_x)
    top = min(landmark_y)
    bottom = max(landmark_y)

    border_list = [left, right, top, bottom]
    return border_list


def landmark_to_points(landmark_x, landmark_y, w, h):
    points = []
    for i in range(len(landmark_x)):
        point = dlib.point(int(landmark_x[i] * w), int(landmark_y[i] * h))
        points.append(point)
    rect = dlib.rectangle(0, 0, int(w), int(h))
    res = dlib.full_object_detection(rect, points)
    # 返回的是full_object_detection类型数据,其中包含的是points的数据,通过这个数据就可以使用
    # dlib.get_face_chip了
    return res


def crop_face_by_landmark_only(img, landmark_x, landmark_y, w, h, face_size):
    res = landmark_to_points(landmark_x, landmark_y, w, h)
    face = dlib.get_face_chip(img, res, size=face_size)
    return face


def generate_split_data(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    content = os.listdir(save_path)
    if len(content) > 0 and content[0] != '.DS_Store':
        print('you had better make the res dir is empty')
        print('your dir is not empty:', content)
        raise NotImplementedError
    dataset_name = ["img", "face", "iwf"]
    emotion_dir = ['Disgust', 'Happy', 'Surprised', 'Sad', 'Angry', 'Neutral', 'Fear']
    spilt_data = ["train", "test"]
    for k in dataset_name:
        if not os.path.exists(os.path.join(save_path, k)):
            os.mkdir(os.path.join(save_path, k))
        for i in spilt_data:
            if not os.path.exists(os.path.join(save_path, k, i)):
                os.mkdir(os.path.join(save_path, k, i))
            for j in emotion_dir:
                if not os.path.exists(os.path.join(save_path, k, i, j)):
                    os.mkdir(os.path.join(save_path, k, i, j))


def generate_dir_res_from_videos(save_path):
    content = os.listdir(save_path)
    if len(content) > 0 and content[0] != '.DS_Store':
        print('you had better make the res dir is empty')
        print('your dir is not empty:', content)
        raise NotImplementedError
    emotion_dir = ['Disgust', 'Happy', 'Surprised', 'Sad', 'Angry', 'Neutral', 'Fear']
    for k in emotion_dir:
        if not os.path.exists(os.path.join(save_path, k)):
            os.mkdir(os.path.join(save_path, k))


def generate_emotion_label(toppath,split):
    for s in split:
        for dir in sorted(os.listdir(os.path.join(toppath,s))):
            if dir != '.DS_Store':
                files = os.listdir(os.path.join(toppath,s, dir))
                emotion = []
                for file in files:
                    if 'annotator' in file.split('.')[0]:
                        F = open(os.path.join(toppath,s, dir, file))
                        emotion.append(F.readline().split(' ')[0])
                        F.close()
                label = max(emotion, key=emotion.count)
                wr = open(os.path.join(toppath,s, dir, dir + '_label.txt'), 'w')
                wr.write(label)
                wr.close()
    print('write emotion label finished ')


def Generate_Data(toppath, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    generate_dir_res_from_videos(save_path)
    dataset_name = ["img", "face", "iwf"]
    for dir in tqdm(sorted(os.listdir(toppath))):
        if dir != '.DS_Store':
            files = os.listdir(os.path.join(toppath, dir))
            dir_path = os.path.join(toppath, dir)
            #  print(dir_path)
            if dir in IGNORE_DATA:
                print('\n ignore dir:', dir)
                continue
            for file in files:
                if file.split('.')[-1] == 'mp4':
                    pre_file_name = file.split('.')[0]
                    # print('process:', pre_file_name)
                    landmark_data = get_landmark_data_from_file(os.path.join(dir_path, pre_file_name + '_landmark.txt'))
                    cap = cv2.VideoCapture(os.path.join(dir_path, pre_file_name + '.mp4'))
                    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 1
                    frames_num = int(cap.get(7))

                    emotion_F = open(os.path.join(dir_path, dir + '_label.txt'))
                    emotion = emotion_F.readline().split(' ')[0]
                    if emotion == 'Unknown':
                        print('Unknown data, re-annotate it please:', dir)
                        continue
                    if emotion == 'Contempt':
                        emotion = 'Disgust'
                    #  print(emotion)
                    for _meaningless in range(2 * fps):
                        sucess, frame = cap.read()
                    for _index, _frame in enumerate(range(2 * fps + 2, frames_num - fps + 1)):
                        success, frame = cap.read()
                        if sucess:
                            if _index > len(landmark_data) - 1:
                                print('len of landmark less than video')
                                break
                            data_list = landmark_data[_index].split(' ')
                            landmark = [float(i) for i in data_list[2:]]
                            landmark_x = landmark[0::2]
                            landmark_y = landmark[1::2]

                            border_list = get_border_list(landmark_x, landmark_y)
                            h, w = frame.shape[0], frame.shape[1]
                            left_top, right_bottom = get_lefttop_and_rightbottom_with_border(landmark, w, h,
                                                                                             border_list)
                            img_without_face = frame
                            img = frame
                            img = cv2.resize(img, (320, 240))
                            face = crop_face_by_landmark_only(frame, landmark_x, landmark_y, w, h, face_size=180)
                            lf_y = 0 if left_top[1] < 0 else left_top[1]
                            rb_y = h if right_bottom[1] > h else right_bottom[1]
                            lf_x = 0 if left_top[0] < 0 else left_top[0]
                            rb_x = w if right_bottom[0] > w else right_bottom[0]
                            img_without_face[lf_y:rb_y, lf_x:rb_x] = 0
                            img_without_face = cv2.resize(img_without_face, (320, 240))

                            if not os.path.exists(os.path.join(save_path, emotion, dir)):
                                os.mkdir(os.path.join(save_path, emotion, dir))
                            for x in dataset_name:
                                if not os.path.exists(os.path.join(save_path, emotion, dir, x)):
                                    os.mkdir(os.path.join(save_path, emotion, dir, x))
                            cv2.imwrite(os.path.join(save_path, emotion, dir, 'img', str(_index).zfill(3) + '.jpg'),
                                        img)
                            cv2.imwrite(os.path.join(save_path, emotion, dir, 'face', str(_index).zfill(3) + '.jpg'),
                                        face)
                            cv2.imwrite(os.path.join(save_path, emotion, dir, 'iwf', str(_index).zfill(3) + '.jpg'),
                                        img_without_face)


# def split_train_test(save_path, toppath, radio=0.7, model='cp'):
#     generate_split_data(save_path)
#     dataset_name = ["img", "face", "iwf"]
#     emotion_path_list = {
#         'Disgust': [],
#         'Happy': [],
#         'Surprised': [],
#         'Sad': [],
#         'Angry': [],
#         'Neutral': [],
#         'Fear': [],
#     }
#     for emotion in emotion_path_list.keys():
#         dir_path = os.path.join(toppath, emotion)
#         for dir in sorted(os.listdir(os.path.join(toppath, emotion))):
#             emotion_path_list[emotion].append(os.path.join(dir_path, dir))
#
#     for k, v in emotion_path_list.items():
#
#         l = int(len(v) * radio)
#         random.shuffle(v)
#         for name in dataset_name:
#             train_count = 0
#             for train_item in v[:l]:
#                 train_cmd = 'mv  ' + os.path.join(train_item, name) + ' ' + os.path.join(save_path, name, 'train',
#                                                                                          k) + '/S' + str(
#                     train_count).zfill(4)
#                 os.system(train_cmd)
#                 train_count += 1
#             test_count = 0
#             for test_item in v[l:]:
#                 test_cmd = 'mv  ' + os.path.join(test_item, name) + ' ' + os.path.join(save_path, name, 'test',
#                                                                                        k) + '/S' + str(
#                     test_count).zfill(4)
#                 test_count += 1
#                 os.system(test_cmd)
#
def split_data(souce_path, save_path, model='train'):
    dataset_name = ["img", "face", "iwf"]
    emotion_path_list = {
        'Disgust': [],
        'Happy': [],
        'Surprised': [],
        'Sad': [],
        'Angry': [],
        'Neutral': [],
        'Fear': [],
    }
    for emotion in emotion_path_list.keys():
        dir_path = os.path.join(souce_path, model, emotion)
        for dir in sorted(os.listdir(dir_path)):
            emotion_path_list[emotion].append(os.path.join(dir_path, dir))

    for k, v in emotion_path_list.items():
        random.shuffle(v)
        for name in dataset_name:
            count = 0
            for item in v:
                cmd = 'mv  ' + os.path.join(item, name) + ' ' + os.path.join(save_path, name, model,
                                                                                         k) + '/S' + str(
                    count).zfill(4)
                os.system(cmd)
                count += 1


def split_train_test_in_ori_data(source_path, target_path, splits):
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    emotion_path_list = {
        'Disgust': [],
        'Happy': [],
        'Surprised': [],
        'Sad': [],
        'Angry': [],
        'Neutral': [],
        'Fear': [],
        'Contempt': []
    }
    for split in splits:
        if not os.path.exists(os.path.join(target_path, split)):
            os.mkdir(os.path.join(target_path, split))
    DATA = os.listdir(source_path)
    random.shuffle(DATA)
    for dir in DATA:
        if dir != '.DS_Store':
            dir_path = os.path.join(source_path, dir)
            F = open(os.path.join(source_path, dir, dir + '_label.txt'))
            emotion_path_list[F.readline().split(' ')[0]].append(dir_path)
    train_count = 0
    test_count = 0
    for k, v in tqdm(emotion_path_list.items()):
        if len(v) < 100:
            radio = 0.6
        else:
            radio = 0.7
        l = int(len(v) * radio)
        random.shuffle(v)

        for train_item in v[:l]:
            sour_train = train_item.split('/')[-1]
            target_train = 'S' + str(train_count).zfill(4)
            train_cmd = 'cp -r ' + train_item + ' ' + os.path.join(target_path, 'Train', target_train)
            os.system(train_cmd)
            for tl in os.listdir(os.path.join(target_path, 'Train', target_train)):
                os.system('mv ' + os.path.join(target_path, 'Train', target_train, tl) + ' ' + os.path.join(target_path,
                                                                                                            'Train',
                                                                                                            target_train,
                                                                                                            tl.replace(
                                                                                                                sour_train,
                                                                                                                target_train)))
            train_count += 1

        for test_item in v[l:]:
            sour_test = test_item.split('/')[-1]
            target_test = 'S' + str(test_count).zfill(4)
            test_cmd = 'cp -r ' + test_item + ' ' + os.path.join(target_path, 'Test', target_test)
            os.system(test_cmd)
            for tl in os.listdir(os.path.join(target_path, 'Test', target_test)):
                os.system('mv ' + os.path.join(target_path, 'Test', target_test, tl) + ' ' + os.path.join(target_path,
                                                                                                          'Test',
                                                                                                          target_test,
                                                                                                          tl.replace(
                                                                                                              sour_test,
                                                                                                              target_test)))
            test_count += 1


def get_distribution(path_n):
    count = {'body': 0, 'interaction': 0, 'other': 0}
    emotion = {'body': {}, 'interaction': {}, 'other': {}}
    xxxx = {}
    for root, dirs, files in os.walk(path_n):
        for dir in dirs:
            file = os.path.join(root, dir, dir + '_annotator1.txt')
            F = open(file)
            for k in F.readlines():
                count[k.rstrip('\n').split(' ')[-1]] += 1
                if k.split(' ')[0] in xxxx.keys():
                    xxxx[k.split(' ')[0]] += 1

                else:
                    xxxx[k.split(' ')[0]] = 1

                if k.split(' ')[0] in emotion[k.rstrip('\n').split(' ')[-1]].keys():
                    emotion[k.rstrip('\n').split(' ')[-1]][k.split(' ')[0]] += 1
                else:
                    emotion[k.rstrip('\n').split(' ')[-1]][k.split(' ')[0]] = 1
    print(count)
    print(emotion)
    print(xxxx)


if __name__ == '__main__':

    # ori_path = '/Users/arthur/fsdownload/New_videos/'
    souce_path = '/Users/arthur/fsdownload/R-ecw'
    temp_path = '/Users/arthur/fsdownload/temp'
    target_path = '/Users/arthur/fsdownload/R-ecw-split'
    split = ['train', 'test']
    # 获得对train 和test划分的数据
    # split_train_test_in_ori_data(ori_path, souce_path, ['train', 'test'])
    # for i in split:
    #     Generate_Data(os.path.join(souce_path, i), os.path.join(temp_path, i))
    generate_split_data(target_path)
    for i in split:
        split_data(temp_path, target_path, i)
# os.system('rm -rf '+temp_path)
# for root, dirs, files in os.walk(path):
#     for dir in dirs:
#         file = os.path.join(root, dir, dir + '_label.txt')
#         F = open(file)
#         for k in F.readlines():
#             if k.split(' ')[0]=='Happy':
#                 cmd1='cp '+os.path.join(root, dir, dir+'_landmark.txt')+' '+os.path.join(s,dir+'.txt')
#                 cmd2 = 'cp ' + os.path.join(root, dir, dir + '.mp4') + ' ' + os.path.join(s, dir + '.mp4')
#                 os.system(cmd1)
#                 os.system(cmd2)

# # fear disgust sad angry surprise netural happy
