from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import cv2
import torch
import uformer as model
from PIL import Image
import numpy as np

#rgb_img = cv2.imread(r'E:\测试数据\dataset\SOTS\hazy\0001_0.8_0.2.jpg',cv2.)
img_hazy = Image.open(r'E:\测试数据\dataset\SOTS\hazy\0001_0.8_0.2.jpg')

img_hazy = img_hazy.resize((640,480), Image.ANTIALIAS).convert('RGB')

rgb_img = np.asarray(img_hazy)
img_hazy = (np.asarray(img_hazy) / 255.0)

input_tensor = torch.from_numpy(img_hazy).float().permute(2, 0, 1).cuda().unsqueeze(0)


dehaze_net = model.Uformer()
dehaze_net = torch.nn.DataParallel(dehaze_net).cuda()

dehaze_net.load_state_dict(torch.load("trained_model/Uformer-SOTS/best.pth"))
#model = resnet50(pretrained=True)
target_layer = model.layer4[-1]
#input_tensor = # Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=dehaze_net, target_layer=target_layer, use_cuda=True)

# If target_category is None, the highest scoring category
# will be used for every image in the batch.
# target_category can also be an integer, or a list of different integers
# for every image in the batch.
target_category = 281

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam)