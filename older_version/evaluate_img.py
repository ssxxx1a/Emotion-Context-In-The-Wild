from model.Image_Model.cen import *
import sys
from model.Image_Model.ori_caer import *
from torch.utils.data import DataLoader

MODEL = {'caer': CAERSNet(), 'cen': CEN(), 'resnet': ResNet()}

def evaluate(data, model1_path, model2_path=None, train_type='resnet'):
    model1 = MODEL[train_type]
    if model2_path is not None:
        model2 = MODEL[train_type]
    if torch.cuda.is_available():
        model1.cuda()
        model1 = nn.DataParallel(model1)
        model1.eval()
        if model2_path is not None:
            model2.cuda()
            model2 = nn.DataParallel(model2)
            model2.eval()
    model1.load_state_dict(torch.load(model1_path))
    # print(list(model1.state_dict().items())[0])
    # print(list(dict_1.items())[0])
    if model2_path is not None:
        model2.load_state_dict(torch.load(model2_path))
    count = 0
    total_count = 0
    vaild_distribution = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
    emtion_label = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}
    keys = list(vaild_distribution.keys())
    emotion_keys = list(emtion_label.keys())
    for i, batch in tqdm(enumerate(data)):
        face = batch['face']
        img = batch['img']
        label = batch['label']
        if torch.cuda.is_available():
            face = face.cuda(non_blocking=True)
            img = img.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
        # print(model1(img))
        # print(model2(face))
        # print(label)
        if train_type == 'resnet':
            if model2_path is not None:
                res2 = model2(face)
            res1 = model1(face)
        else:
            res1 = model1(img, face)
        if model2_path is not None:
            res = res2 + res1
            res = F.softmax(res)

        else:
            res = res1
        for xx in range(len(res)):
            if int(res[xx].argmax().item()) == int(label[xx].item()):
                count += 1
                vaild_distribution[str(label[xx].item())] += 1

        total_count += len(label)
        total_acc = count / total_count
        sys.stdout.write(
            "\r[Batch %d/%d][example ~ argamx_index:%d label:%d][total_acc:%f]\n"
            % (
                i,
                len(vaild_data),
                int(res[0].argmax().item()),
                int(label[0]),
                total_acc
                #  str(res)
            )
        )
    for i in range(len(vaild_distribution)):
        vaild_distribution.update({emotion_keys[i]: vaild_distribution.pop(keys[i])})

    print(vaild_distribution)


def ssss(data, model_path):
    model = ResNet()
    param_face_dict = torch.load(model_path)
    face_dict = model.state_dict()
    pretrained_face_dict = {k.strip('module.'): v for k, v in param_face_dict.items()
                            if
                            k.strip('module.') in face_dict.keys()}
    print('Total : {}, update: {}'.format(len(param_face_dict), len(pretrained_face_dict)))
    face_dict.update(pretrained_face_dict)
    model.load_state_dict(face_dict)
    model.eval()
    # print(list(param_face_dict.items())[-2])
    # print(list(model.state_dict().items())[-2])
    d = next(iter(data))
    img = d['img']
    label = d['label']
    face = d['face']
    print(F.softmax(model(img), dim=1))
    print(label)

model_path = '/Users/arthur/fsdownload/res/9.9_res/No.10/FER_resnet_img_50.pth'
if __name__ == '__main__':
    vaild_data = DataLoader(
        EmotionDataset('/opt/data/private/dbmeng/Data/Emotion/Caer/Caer-S/test', txt_path='../data/label_file/test.txt', img_size=224,
                       face_size=96, IsMark=False),
        batch_size=32,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=16,
        collate_fn=my_collate_fn
    )

    backbone_face_path = '/opt/data/private/pycharm_map/Context-emotion/saved_model/FER_resnet_face_50.pth'
    backbone_img_path = '/opt/data/private/pycharm_map/Context-emotion/saved_model/FER_resnet_img_50.pth'
    evaluate(vaild_data, backbone_face_path)
    # ssss(vaild_data, backbone_img_path)
    a={'Angry': 2988, 'Disgust': 2987, 'Fear': 2986, 'Happy': 2995, 'Neutral': 2987, 'Sad': 2988, 'Surprise': 2982}
  #  b={'Angry': 2644, 'Disgust': 2876, 'Fear': 2942, 'Happy': 2442, 'Neutral': 2235, 'Sad': 2743, 'Surprise': 2652}
    b={'Angry': 2038, 'Disgust': 2407, 'Fear': 2681, 'Happy': 2298, 'Neutral': 1556, 'Sad': 2283, 'Surprise': 2085}
    #b={'Angry': 1865, 'Disgust': 2322, 'Fear': 2590, 'Happy': 2137, 'Neutral': 1345, 'Sad': 2509, 'Surprise': 1903}
    res={}
    ave=0
    for k,v in a.items():
        res[k]=round(b[k]/v,4)
        ave+=res[k]
    ave/=7
    print(res)
    print(ave)
