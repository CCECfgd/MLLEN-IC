import torch
import torch.nn as nn
import torch.optim
import os
import argparse
import time
#import wave_u_net as model
#import uformer as model
import Res2Net as model
import dataloader
from SSIM import SSIM
import torchvision
import matplotlib.pyplot as plt
import VGG
import numpy as np

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        #init.xavier_uniform(m.weight)
        m.weight.data.normal_(0, 0.02)
        #m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        #init.xavier_uniform(m.weight)
        m.weight.data.normal_(0, 0.02)
        #m.bias.data.zero_()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def train(config):

    dehaze_net = model.U_Net_SE_RES(planes=[64,128,256,512])
    dehaze_net = nn.DataParallel(dehaze_net).cuda()

    #dehaze_net.apply(inplace_relu)
    dehaze_net.apply(initialize_weights)
    #dehaze_net.load_state_dict(torch.load("trained_model/New_U_Net_SE_RES_SKIP_1-SSIM-LOL/best.pth"))
    train_dataset = dataloader.dehazing_loader(config.orig_images_path,config.hazy_images_path)
    val_dataset = dataloader.dehazing_loader(config.orig_images_path_val,config.hazy_images_path_val, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)
    #定义感知损失
    vgg_loss = VGG.PerceptualLoss()
    vgg_loss.cuda()
    vgg = VGG.load_vgg16("./model")
    vgg.eval()
    for param in vgg.parameters():
        param.requires_grad = False



    L2 = nn.MSELoss()

    L1  = nn.SmoothL1Loss()


    criterion = SSIM()
    comput_ssim = SSIM()
    
    dehaze_net.train()

    zt = 1
    Iters = 0
    iter_loss = []#计数损失曲线用
    indexX = []                                                                                                   #计数损失曲线用
    indexY = []
    #config.lr = 0.0003

    for epoch in range(0, config.num_epochs):

        if epoch <= 1:
            config.lr = 0.0003
        elif epoch > 1 and epoch <= 15:
            config.lr = 0.0001
        elif epoch > 15 and epoch <= 30:
            config.lr = 0.00006
        elif epoch > 30 and epoch <= 40:
            config.lr = 0.00003
        elif epoch > 40 and epoch <= 50:
            config.lr = 0.00001
        elif epoch > 50 and epoch <= 60:
            config.lr = 0.000006
        elif epoch > 60 and epoch <= 70:
            config.lr = 0.000003
        elif epoch > 70 and epoch <= 200:
            config.lr = 0.000001
        print("now lr == %f"%config.lr)
        print("*" * 80 + "第%i轮" % epoch + "*" * 80)
        optimizer = torch.optim.AdamW(dehaze_net.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=0.02)
        for iteration, (img_clean, img_haze, att) in enumerate(train_loader):

            img_clean = img_clean.cuda()
            img_haze = img_haze.cuda()
            att = att.cuda()

            try:
                clean_image = dehaze_net(img_haze,att)
                if config.lossfunc == "MSE":
                    loss = criterion(clean_image, img_clean)                                                             # MSE损失
                elif config.lossfunc == 'L1':
                    loss  = criterion(clean_image, img_clean)
                else:
                    ssimloss = criterion(img_clean, clean_image)                                                            # -SSIM损失
                    ssimloss = 1-ssimloss
                    vggloss = vgg_loss.compute_vgg_loss(vgg, clean_image, img_clean)
                    #loss = 0.001*vggloss
                    #print("VGG loss:",vggloss)
                    l1loss = L1(clean_image, img_clean)
                    # if epoch<5:
                    #     vggloss = 0
                    #     loss = loss
                    # else:
                    #     loss = 1 * loss + 0.0001 * vggloss
                    loss = 1 * ssimloss + 0.0001 * vggloss
                    #print(0.5* loss)
                    # print(0.0001*vggloss)
                    #loss = 0.0005*vggloss +  l1loss
                # indexX.append(loss.item())
                # indexY.append(iteration)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(dehaze_net.parameters(), config.grad_clip_norm)
                optimizer.step()
                #Iters += 1
                #if ((iteration + 1) % config.display_iter) == 0:
                iter_loss.append(loss.item())
                print("Loss at", iteration, ":%0.6f"% loss.item(),'VGG-loss:%i'%vggloss.item(),"SSIM-loss:","%0.6f"%ssimloss.item())
                # if ((iteration + 1) % config.snapshot_iter) == 0:
                #     torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
                Iters+=1
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e
        #TODO:保存np.array到本地
        _ssim=[]
        #torch.save(dehaze_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
        print("start Val!")
        #Validation Stage
        with torch.no_grad():
            for iteration1, (img_clean, img_haze,img_att) in enumerate(val_loader):
                print("va1 : %s"%str(iteration1))
                img_clean = img_clean.cuda()
                img_haze = img_haze.cuda()
                img_att = img_att.cuda()


                clean_image = dehaze_net(img_haze, img_att)
                
                _s = comput_ssim(img_clean,clean_image)
                _ssim.append(_s.item())
                torchvision.utils.save_image(torch.cat((img_haze,img_clean,clean_image), 0),config.sample_output_folder + "/epoch%s" % epoch +"/"+ str(iteration1 + 1) + ".jpg")
                #torchvision.utils.save_image(clean_image,config.sample_output_folder + "/epoch%s" % epoch + "/" + str(iteration1 + 1) + ".jpg")
            _ssim = np.array(_ssim)
            
            print("-----The %i Epoch mean-ssim is :%f-----" %(epoch,np.mean(_ssim)))

            with open("trainlog/U_Net_SE_RES_1-att-%s%s2-LOL.log" % (config.lossfunc,config.actfuntion), "a+", encoding="utf-8") as f:
                s = "[%i,%f]" %(epoch,np.mean(_ssim))+ "\n"
                f.write(s)

            indexX.append(epoch)
            now = np.mean(_ssim)
            if indexY == []:
                indexY.append(now)
                print("First epoch，Save！")
                torch.save(dehaze_net.state_dict(), config.snapshots_folder + 'best.pth')
                print('saved first pth!')
            else:
                now_max = np.argmax(indexY)
                indexY.append(now)
                print('max epoch %i' % now_max,'SSIM:',indexY[now_max],'Now Epoch mean SSIM is:', now)

                if now >= indexY[now_max]:
                    torch.save(dehaze_net.state_dict(), config.snapshots_folder  + 'best.pth')
                    print('saved pth!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #print(indexX,indexY)
        plt.plot(indexX,indexY,linewidth=2)
        plt.pause(0.1)
    plt.savefig("trainlog/U_Net_SE_RES_1-att-%s2-LOL.png" % config.lossfunc)
    torch.save(dehaze_net.state_dict(), config.snapshots_folder + "uformer.pth")

if __name__ == "__main__":
    """
    	当前，输入为有雾图像和无雾图像，有雾图像和无雾图像都转为灰度图进人网络
    	:param orig_images_path:深度图下采样，当前用有雾图像的灰度图代替
    	:param hazy_images_path:有雾彩色图像
    	:param label_images_path:深度图，用无雾图像灰度图代替
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--orig_images_path', type=str, default=r"E:\低光照增强相关文件\部分低光照增强数据集\LOL\train\gt\\")
    parser.add_argument('--hazy_images_path', type=str, default=r"E:\低光照增强相关文件\部分低光照增强数据集\LOL\train\dark\\")


    parser.add_argument('--orig_images_path_val', type=str, default=r"E:\低光照增强相关文件\部分低光照增强数据集\LOL\val\gt\\")
    parser.add_argument('--hazy_images_path_val', type=str, default=r"E:\低光照增强相关文件\部分低光照增强数据集\LOL\val\dark\\")


    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=1)
    parser.add_argument('--snapshot_iter', type=int, default=50)
    parser.add_argument('--snapshots_folder', type=str, default="trained_model/U_Net_SE_RES_VGG-att-LOL/")
    parser.add_argument('--sample_output_folder', type=str, default="sample/U_Net_SE_RES_VGG-att-LOL")
    parser.add_argument('--in_or_out', type=str, default="outdoor")
    parser.add_argument('--lossfunc', type=str, default="SSIM",help="choose Loss Function(MSE or -SSIM-b1-n10-aodnet).")
    parser.add_argument('--actfuntion', type=str, default="relu",help="drelu or relu")
    parser.add_argument('--cudaid', type=str, default="0",help="choose cuda device id 0-7).")

    config = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = config.cudaid

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    if not os.path.exists(config.sample_output_folder):
        os.mkdir(config.sample_output_folder)

    for i in range(config.num_epochs):
        path = config.sample_output_folder + "/epoch%s" % str(i)
        if not os.path.exists(path):
            os.mkdir(path)

    s = time.time()
    train(config)
    e = time.time()
    print(str(e-s))
