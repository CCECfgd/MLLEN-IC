import cv2
import numpy as np
from PIL import Image
#图像标准化 求255-I

def att(channal):
    cv2.normalize(channal, channal, 0, 255, cv2.NORM_MINMAX)
    M = np.ones(channal.shape, np.uint8) * 255
    img_new = cv2.subtract(M, channal)
    return img_new
def process(img):
    b, g, r = cv2.split(img)
    b = att(b)
    g = att(g)
    r = att(r)
    new_image = cv2.merge([b, g, r])
    return new_image


image = np.array(Image.open(r'E:\dataset\O-HAZE\test\hazy\45_outdoor_hazy.jpg').convert('RGB').resize((512,512)))
new_image = process(image)
cv2.imshow('new_image',new_image)
#image = cv2.imread('E:\低光照增强相关文件\低光照数据集\VV\P1000333.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# normalized_image = cv2.normalize(gray,None)
cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)

M=np.ones(gray.shape,np.uint8)*255
img_new=cv2.subtract(M,gray)


cv2.waitKey(0)
cv2.destroyAllWindows()

