from torchvision import models, transforms
import argparse
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
from model.Video_Model.caen import *

class FeatureExtraction():
	def __init__(self, model, target_layers='attention_inference_module'):
		self.model = model
		self.target_layers = target_layers
		self.gradients = [0]

	def save_gradient(self, grad):
		self.gradients[0] = grad

	def get_gradients(self, ind):
		return self.gradients[ind]

	def __call__(self, img,face):
		for name, module in self.model._modules.items():
			print(name,module)
			# if name == 'fc':
			# 	x = x.view(x.size(0), -1)
			if name=='backbone_img':
				img = module(img)
			if name=='backbone_face':
				face = module(face)
			if name=='attention_inference_module':
				res=module(img,face)

			if name in self.target_layers:
				res.register_hook(self.save_gradient)
				target_features = res

			print("Layer Name: ", name)
		return target_features, res


class GradCam:
	def __init__(self, model, target_layer_names):
		self.model = model
		self.extractor = FeatureExtraction(self.model, target_layer_names)

	# def forward(self, input_):
	# 	return self.model(input_)

	def __call__(self, img,face, index=None):
		_, _, H, W = img.shape
		targetFeatures, output = self.extractor(img,face)

		if index is None:
			index = torch.argmax(output, 1).squeeze().cpu().numpy()

		one_hot = torch.max(output)
		self.model.zero_grad()
		one_hot.backward()

		grads_val = self.extractor.get_gradients(0)
		weights = torch.mean(grads_val.squeeze(), dim=(1, 2))

		weights = torch.reshape(weights, (-1, 1, 1))
		cam = torch.sum(torch.mul(targetFeatures.squeeze(), weights), dim=0)
		cam = cam.detach().cpu().numpy()

		cam = np.maximum(cam, 0) # ReLU
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		outCAM = cv2.resize(np.uint8(cam * 255), (W, H))
		index='attention'
		return index, outCAM

def LoadNet(modelpath):
	net = models.resnet34(pretrained=False)
	net.load_state_dict(torch.load(modelpath))
	net.eval()
	net.cuda()
	return net


normMean = [0.485, 0.456, 0.406]
normStd = [0.229, 0.224, 0.225]
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(normMean, normStd)])


# save CAM to savepth
def saveCAM(imgUrl, savepath, pred, CAM):
	if not os.path.exists(savepath):
		os.mkdir(savepath)

	name = imgUrl.split('/')[-1]
	savename = name.split('.')[0] + '_' + str(pred)
	print(savename)
	img = cv2.imread(imgUrl, 1)
	img=cv2.resize(img,(224,224))
	attentionmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
	overlap = attentionmap * 0.5 + img * 0.5

	cv2.imwrite(os.path.join(savepath, savename + '_GradCAM.png'), attentionmap)
	cv2.imwrite(os.path.join(savepath, savename + '_overlap.png'), overlap)


def GetArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-M', '--modelPth', type=str, default='./data/resnet34-333f7ec4.pth',
						help='the network model path')

	parser.add_argument('-P', '--readImgUrl', type=str, default='/opt/data/private/pycharm_map/Context-emotion/data/plane.jpeg',
						help='Url of the image for attention map visualization')

	parser.add_argument('-S', '--savePth', type=str, default='data/results/GradCAM/',
						help='the path to save attention maps and overlapped images')
	arg = parser.parse_args()
	return arg


if __name__ == '__main__':
	Args = GetArgs()
	device='cuda' if torch.cuda.is_available() else 'cpu'

	img = cv2.imread(Args.readImgUrl, 1)
	# img = cv2.resize(img, (224, 224))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	H, W, _ = img.shape
	torch_img = preprocess(img).unsqueeze(0).to(device)
	net=CAEN()
	net.load_state_dict(torch.load('/opt/data/private/pycharm_map/Context-emotion/log/caen_epoch-19.pth.tar'))
	grad_cam = GradCam(model=net, target_layer_names=["attention_inference_module"])
	# net = LoadNet(Args.modelPth)  # load model
	# grad_cam = GradCam(model=net, target_layer_names=["layer4"])
	#
	# # None for the highest scoring category; or targets the requested index.
	# Prediction, CAMap = grad_cam(torch_img)
	# print(Args.readImgUrl, ' finished generation !')
	# # print(grad_cam.forward(torch_img))
	#
	# saveCAM(Args.readImgUrl, Args.savePth, Prediction, CAMap)  # save CAM and overlap
