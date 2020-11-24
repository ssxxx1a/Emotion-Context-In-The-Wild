import os
import sys
from tqdm import tqdm
import cv2
import dlib
import random

# detect face
detector = dlib.get_frontal_face_detector()
predicter_path = '/opt/data/private/pycharm_map/Context-emotion/shape_predictor_68_face_landmarks.dat'
sp = dlib.shape_predictor(predicter_path)

def Generate_data_txt(root, txt_path, is_face=True):
    emtion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
    format = ['jpg', 'png', 'jepg']
    file = open(txt_path, mode='w')
    if not is_face:
        for root, dirs, files in os.walk(root):
            for i in files:
                if i.split('.')[-1].lower() in format:
                    imgpath = os.path.join(root, i)
                    label = imgpath.split('/')[-2]
                    if label not in emtion:
                        continue
                    img = cv2.imread(imgpath)
                    try:
                        face = detector(img, 1)
                    except:
                        continue
                    if len(face) > 0:
                        face = face[0]
                        face_area = [face.top(), face.bottom(), face.left(), face.right()]
                        info = imgpath + ':' + str(face_area) + ':' + label + '\n'
                        print('write file:', imgpath)
                        file.write(info)
                    else:
                        #os.system('rm -rf ' + imgpath)
                        print('error img')
                        continue
    else:
        for root, dirs, files in os.walk(root):
            for i in files:
                if i.split('.')[-1].lower() in format:
                    imgpath = os.path.join(root, i)
                    label = imgpath.split('/')[-2]
                    info = imgpath + ':' + str(label) + '\n'
                    print('write file:', info)
                    file.write(info)
    print('finished')
    file.close()


def copy_img_data(root, target_path, count_num=1000):
    format = ['jpg', 'png', 'jepg']
    for root, dirs, files in os.walk(root):
        for i in dirs:
            if not os.path.exists(os.path.join(target_path + i)):
                print('create dir:', i)
                os.mkdir(os.path.join(target_path, i))
        count = 0
        if len(files) != 0:
            random.shuffle(files)
        else:
            continue
        print('copy data....')
        for i in files:
            path = os.path.join(root, i)
            emotion = path.split('/')[-2]
            if path.split('/')[-1].split('.')[-1].lower() in format:
                pwd = 'cp ' + path + ' ' + target_path + '/' + emotion
                os.system(pwd)
            count += 1
            if count == count_num:
                print('cp num:', count)
                break


def cropFace(root, target_path, Use_Count=False):
    format = ['jpg', 'png', 'jepg']
    for root, dirs, files in os.walk(root):
        for i in dirs:
            if not os.path.exists(os.path.join(target_path + i)):
                print('create dir:', i)
                os.mkdir(os.path.join(target_path, i))
        print('crop face and save....')
        if Use_Count:
            count = 0
        for i in files:
            path = os.path.join(root, i)
            emotion = path.split('/')[-2]
            if path.split('/')[-1].split('.')[-1].lower() in format:
                img = cv2.imread(path)
                face_area = detector(img, 1)
                if len(face_area):
                    face_area = face_area[0]
                    # faces = dlib.full_object_detections()
                    # faces.append(sp(img,face_area))
                    face = dlib.get_face_chip(img, sp(img, face_area), size=224)
                    cv2.imwrite(os.path.join(target_path, emotion, i), face)
                    print('crop face: ', os.path.join(target_path, emotion, i))
                    # face.save(os.path.join(target_path,i))
            if Use_Count:
                count += 1
                if count == 200:
                    break


if __name__ == '__main__':
    # val_root = '/opt/data/private/data/Caer/Caer-S/train'
    # val_target_path = '/opt/data/private/data/Caer/minCaer/Train/'
    # copy_img_data(val_root, val_target_path)
    val_target_path = '/opt/data/private/data/Caer/minCaer/Face/test'
    txt_path = '/opt/data/private/pycharm_map/Context-emotion/crop_face_test.txt'
    Generate_data_txt(val_target_path, txt_path)
    #crop face for train:
    #cropFace('/opt/data/private/data/Caer/Caer-S/train', '/opt/data/private/data/Caer/minCaer/Face/train/')
    #crop face for test
    #cropFace('/opt/data/private/data/Caer/Caer-S/test','/opt/data/private/data/Caer/minCaer/Face/test/',Use_Count=True)
