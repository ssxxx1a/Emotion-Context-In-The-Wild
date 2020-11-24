'''
save the CAM graph ,but has bug in this file
'''
from model.Additional_model.CAM.GradCAM import *
from model.Video_Model.caen import *

if __name__ == '__main__':
    Args = GetArgs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = torch.rand(size=(1, 3, 224, 224)).cuda()
    face = torch.rand(size=(1, 3, 112, 112)).cuda()
    net = CAEN().cuda()
    grad_cam = GradCam(model=net, target_layer_names=["attention_inference_module"])
    Prediction, CAMap = grad_cam(img, face)
    print(Args.readImgUrl, ' finished generation !')
    # print(grad_cam.forward(torch_img))

    saveCAM(Args.readImgUrl, Args.savePth, Prediction, CAMap)  # save CAM and overlap


# net = LoadNet(Args.modelPth)  # load model
# grad_cam = GradCam(model=net, target_layer_names=["layer4"])
#
# # None for the highest scoring category; or targets the requested index.
# Prediction, CAMap = grad_cam(torch_img)
# print(Args.readImgUrl, ' finished generation !')
# # print(grad_cam.forward(torch_img))
#
# saveCAM(Args.readImgUrl, Args.savePth, Prediction, CAMap)  # save CAM and overlap
