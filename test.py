import torch
import torchvision
import torch.optim
import Res2Net as model
import numpy as np
import torch.nn as nn
from PIL import Image
import glob
import os,cv2

def att(channal):
    cv2.normalize(channal, channal, 0, 255, cv2.NORM_MINMAX)
    M = np.ones(channal.shape, np.uint8) * 255
    img_new = cv2.subtract(M, channal)
    return img_new
def process(img):
    #img = np.array(img)
    b, g, r = cv2.split(img)
    b = att(b)
    g = att(g)
    r = att(r)
    new_image = cv2.merge([r,g,b])
    return new_image

def TEST(image_hazy_path):

    LL_image = np.array(Image.open(image_hazy_path).convert('RGB'))

    if LL_image.shape[0] > 2000:
        index = int(LL_image.shape[1]/2),int(LL_image.shape[0]/2)
        LL_image = cv2.resize(LL_image, index)

    if LL_image.shape[0] % 16 != 0 or LL_image.shape[1] % 16 != 0:
        i = LL_image.shape[0]
        j = LL_image.shape[1]
        if LL_image.shape[0] % 16 != 0:
            while i % 16 != 0:
                i-=1
        if LL_image.shape[1]% 16 != 0:
            while j % 16 != 0:
                j -= 1

        LL_image = cv2.resize(LL_image, (j,i))

    attention = process(LL_image)

    LL_image = (np.asarray(LL_image) / 255.0)

    attention = attention / 255.0

    LL_image = torch.from_numpy(LL_image).float().permute(2, 0, 1).cuda().unsqueeze(0)

    attention = torch.from_numpy(attention).float().permute(2, 0, 1).cuda().unsqueeze(0)

    with torch.no_grad():
        enImage = LLIENET(LL_image,attention)

    index = image_hazy_path.split('/')[-1]

    torchvision.utils.save_image(enImage, "output/%s" % (index))
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    pth_path = "weights/best.pth"
    LLIENET = model.U_Net_SE_RES(planes=[64,128,256,512])

    LLIENET = nn.DataParallel(LLIENET).cuda()
    LLIENET.load_state_dict(torch.load(pth_path))

    LLI_list = glob.glob("input/*")
    print('image num:',len(LLI_list))
    for Id in range(len(LLI_list)):
        TEST(LLI_list[Id])
        print(LLI_list[Id], "done!")