import torch
import torchvision
import torch.optim
import Res2Net as model
import numpy as np
import torch.nn as nn
from PIL import Image
import glob
import time,os,cv2

def att(channal):
    cv2.normalize(channal, channal, 0, 255, cv2.NORM_MINMAX)
    M = np.ones(channal.shape, np.uint8) * 255
    img_new = cv2.subtract(M, channal)
    return img_new
def process(img):
    img = np.array(img)
    b, g, r = cv2.split(img)
    b = att(b)
    g = att(g)
    r = att(r)
    new_image = cv2.merge([b, g, r])
    return new_image

def dehaze_image(image_hazy_path,Id,spath,pth_path):


    img_hazy = Image.open(image_hazy_path).convert('RGB')
    attention = process(img_hazy)
    if img_hazy.size[0] > 2000:


        index = int(img_hazy.size[0]/2),int(img_hazy.size[1]/2)
        img_hazy = img_hazy.resize(index, Image.ANTIALIAS)
    if img_hazy.size[0] % 16 != 0 or img_hazy.size[1] % 16 != 0:
        i = img_hazy.size[0]
        j = img_hazy.size[1]
        if img_hazy.size[0] % 16 != 0:
            while i % 16 != 0:
                i-=1
        if img_hazy.size[1]% 16 != 0:
            while j % 16 != 0:
                j -= 1
        img_hazy = img_hazy.resize((i,j), Image.ANTIALIAS)
    attention = process(img_hazy)

    img_hazy = (np.asarray(img_hazy) / 255.0)
    attention = attention / 255.0



    img_hazy = torch.from_numpy(img_hazy).float().permute(2, 0, 1).cuda().unsqueeze(0)
    attention = torch.from_numpy(attention).float().permute(2, 0, 1).cuda().unsqueeze(0)

   

    with torch.no_grad():
        clean_image = dehaze_net(img_hazy,attention)

    temp_tensor = (clean_image,0)
    index = image_hazy_path.split('\\')[-1]
    
    #torchvision.utils.save_image(torch.cat((img_hazy,clean_image),0), "test_result/real/%s/%s" % (s,index))

    torchvision.utils.save_image(clean_image, "LLE/LOL(base+se+res+att+vgg)-NPE/%s" % (index))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    spath = 'indoor'
    pth_path = "trained_model/U_Net_SE_RES_VGG-att-LOL/best.pth"
    dehaze_net = model.U_Net_SE_RES(planes=[64,128,256,512])


    dehaze_net = nn.DataParallel(dehaze_net).cuda()
    dehaze_net.load_state_dict(torch.load(pth_path))


    testset = 'ITS'

    hazy_list = glob.glob(r"E:\低光照增强相关文件\低光照数据集\NPE\*"  )

    for Id in range(len(hazy_list)):
        dehaze_image(hazy_list[Id],Id,spath,pth_path)
        print(hazy_list[Id], "done!")
   
