import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2
random.seed(1143)

def populate_train_list(orig_images_path, hazy_images_path):


    image_list_haze_index = os.listdir(hazy_images_path)
    image_dataset = []
    for i in image_list_haze_index:
        image_name = i.split('_')[0]+'.jpg'
        image_dataset.append((orig_images_path + image_name, hazy_images_path + i))

    train_list = image_dataset


    return train_list


# def populate_train_list(orig_images_path, hazy_images_path, ):
#      """1to1"""
#     image_list_haze_index = os.listdir(hazy_images_path)
#     image_dataset = []
#     for i in image_list_haze_index:
#
#         image_dataset.append((orig_images_path + i, hazy_images_path + i))
#
#     train_list = image_dataset
#
#     return train_list

# def populate_val_list(orig_images_path, hazy_images_path,):
#
#
#     image_list_haze_index = os.listdir(hazy_images_path)  # 文件名
#     image_dataset = []
#     for i in image_list_haze_index:  # 添加路径，并组合为元组
#         image_dataset.append((orig_images_path + i, hazy_images_path + i))
#
#     val_list = image_dataset
#
#     return val_list

def populate_val_list(orig_images_path, hazy_images_path):


    image_list_haze_index = os.listdir(hazy_images_path)
    image_dataset = []
    for i in image_list_haze_index:
        image_name = i.split('_')[0]+'.png'
        image_dataset.append((orig_images_path + image_name, hazy_images_path + i))

    train_list = image_dataset


    return train_list
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
class dehazing_loader(data.Dataset):

    def __init__(self, orig_images_path, hazy_images_path, mode='train'):

        self.train_list = populate_train_list(orig_images_path, hazy_images_path)
        self.val_list = populate_val_list(orig_images_path, hazy_images_path)

        if mode == 'train':
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):

        data_clean_path, data_hazy_path = self.data_list[index]
        data_clean = Image.open(data_clean_path).convert('RGB')
        data_hazy = Image.open(data_hazy_path).convert('RGB')



        data_clean = data_clean.resize((600,400), Image.ANTIALIAS)
        data_hazy = data_hazy.resize((600,400), Image.ANTIALIAS)
        attention = process(data_hazy)



        data_clean = np.asarray(data_clean) / 255.0
        data_hazy = np.asarray(data_hazy) / 255.0
        attention = attention/ 255.0





        data_clean = torch.from_numpy(data_clean).float()
        data_hazy = torch.from_numpy(data_hazy).float()
        attention = torch.from_numpy(attention).float()




        return data_clean.permute(2, 0, 1), data_hazy.permute(2, 0, 1), attention.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

