import torch
import torchvision


def unnorm_save(img, size, fp, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    t_mean = torch.FloatTensor(mean).view(3, 1, 1).expand(3, size, size)
    t_std = torch.FloatTensor(std).view(3, 1, 1).expand(3, size, size)
    img_GT = img * t_std + t_mean
    img_GT = torchvision.transforms.ToPILImage()(img_GT).convert('RGB')
    img_GT.save("./img/" + fp + ".png")
