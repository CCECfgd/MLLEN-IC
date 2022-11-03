import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
class Pool(nn.Module):
    def __init__(self):
        super(Pool, self).__init__()
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.avgepool = nn.AvgPool2d(2, stride=2, padding=0)

    def forward(self,x):
        return self.maxpool(x)+self.avgepool(x)

class PSA(nn.Module):

    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S

        self.convs = []
        for i in range(S):
            self.convs.append(nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))

        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))

        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()

        # Step1:SPC module
        SPC_out = x.view(b, self.S, c // self.S, h, w)  # bs,s,ci,h,w
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])

        # Step2:SE weight
        se_out = []
        for idx, se in enumerate(self.se_blocks):
            se_out.append(se(SPC_out[:, idx, :, :, :]))
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand_as(SPC_out)

        # Step3:Softmax
        softmax_out = self.softmax(SE_out)

        # Step4:SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)

        return PSA_out


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class Res2NetBottleneck(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=False,  norm_layer=None):
        super(Res2NetBottleneck, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out
class Up_block(nn.Module):
    def __init__(self,in_c,out_c):
        super(Up_block, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        #self.conv1 = Res2NetBottleneck(in_c, in_c)#基础型
        self.conv1 = Res2NetBottleneck_se(in_c, in_c)
        self.conv2 = nn.Sequential(nn.Conv2d(in_c,out_c,3,1,1), nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_c*2,out_c,3,1,1),  nn.LeakyReLU(inplace=True))

    def forward(self,x, skip):
        x = self.up(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.cat((x,skip),1)
        out = self.conv3(x)
        return out

class Up_block2(nn.Module):
    def __init__(self,in_c,out_c):
        super(Up_block2, self).__init__()
        self.deconv =nn.Sequential(nn.ConvTranspose2d(in_c,out_c,2,2,0), nn.LeakyReLU(inplace=True))
        self.conv1 = Res2NetBottleneck(out_c, out_c)

        self.conv2 = nn.Sequential(nn.Conv2d(out_c*2,out_c,3,1,1), nn.LeakyReLU(inplace=True))

    def forward(self,x, skip):
        x = self.deconv(x)
        x = self.conv1(x)
        x = torch.cat((x,skip),1)
        out = self.conv2(x)
        return out
class Res2NetBottleneck_se(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=True,  norm_layer=None):
        super(Res2NetBottleneck_se, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.se = SEModule(planes * self.expansion) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class Res2NetBottleneck_psa(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, stride=1, scales=4, groups=1, se=True,  norm_layer=None):
        super(Res2NetBottleneck_psa, self).__init__()
        if planes % scales != 0:
            raise ValueError('Planes must be divisible by scales')
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d(planes * self.expansion)
        bottleneck_planes = groups * planes
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList([conv3x3(bottleneck_planes // scales, bottleneck_planes // scales, groups=groups) for _ in range(scales-1)])
        self.bn2 = nn.ModuleList([norm_layer(bottleneck_planes // scales) for _ in range(scales-1)])
        self.conv3 = conv1x1(bottleneck_planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.se = PSA(channel=planes * self.expansion,reduction=8)() if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(xs[s])
            elif s == 1:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s]))))
            else:
                ys.append(self.relu(self.bn2[s-1](self.conv2[s-1](xs[s] + ys[-1]))))
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class Resblock(nn.Module):
    def __init__(self,in_c,out_c):
        super(Resblock, self).__init__()
        self.shape = False
        if in_c != out_c:
            self.shape = True
            self.transconv = nn.Conv2d(in_c,out_c,3,1,1)
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, 1, 1),
                nn.LeakyReLU(),
            )
        else:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.LeakyReLU(),
            )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c,out_c,3,1,1),
            nn.LeakyReLU(),
        )

        self.act = nn.LeakyReLU()

    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        if self.shape:
            x = self.transconv(x)
        out = self.act(conv2+x)

        return out

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(Block, self).__init__()
        self.same_shape = same_shape
        if not same_shape:
            strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)
    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out+x)


class U_Net_SE_RES(nn.Module):


    def __init__(self,planes):
        super(U_Net_SE_RES,self).__init__()
        self.planes_1, self.planes_2, self.planes_3, self.planes_4 = planes[0], planes[1], planes[2], planes[3]
        self.encoder()
        self.decoder()

    def encoder(self):
        self.Maxpool = Pool()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(6, self.planes_1, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            Res2NetBottleneck_se(self.planes_1 , self.planes_1)
        )


        self.conv_2 = nn.Sequential(
            nn.Conv2d(self.planes_1 , self.planes_2,3, 1, 1),
            nn.LeakyReLU(inplace=True),
            Res2NetBottleneck_se(self.planes_2 , self.planes_2)
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(self.planes_2 , self.planes_3,3, 1, 1),
            nn.LeakyReLU(inplace=True),
            Res2NetBottleneck_se(self.planes_3 , self.planes_3)
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(self.planes_3 , self.planes_4,3, 1, 1),
            nn.LeakyReLU(inplace=True),
            Res2NetBottleneck_se(self.planes_4 , self.planes_4)
        )

        self.res = nn.Sequential(
            Res2NetBottleneck(self.planes_4 , self.planes_4),
            Res2NetBottleneck(self.planes_4 , self.planes_4),
            Res2NetBottleneck(self.planes_4 , self.planes_4),
            Res2NetBottleneck(self.planes_4 , self.planes_4),
            Res2NetBottleneck(self.planes_4 , self.planes_4),
        )
    def decoder(self):
        self.deconv_1 = Up_block(self.planes_4 , self.planes_3)
        self.deconv_2 = Up_block(self.planes_3 , self.planes_2)
        self.deconv_3 = Up_block(self.planes_2 , self.planes_1)
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(self.planes_1, 3, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )


    def forward(self, input, att):

        # ----------encoder--------------
        conv_1 = self.conv_1(torch.cat((input,att),1))
        pool_1 = self.Maxpool(conv_1)

        conv_2 = self.conv_2(pool_1)
        pool_2 = self.Maxpool(conv_2)

        conv_3 = self.conv_3(pool_2)
        pool_3 = self.Maxpool(conv_3)

        conv_4 = self.conv_4(pool_3)

        res = self.res(conv_4)
        # ----------decoder--------------
        deconv_1 = self.deconv_1(res,conv_3)

        deconv_2 = self.deconv_2(deconv_1,conv_2)

        deconv_3 = self.deconv_3(deconv_2,conv_1)

        deconv_4 = self.deconv_4(deconv_3)

        result = deconv_4.mul(att)
        result = result + input
        return result

class Flash64(nn.Module):
    def __init__(self):
        super(Flash64, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6,64,3,1,1),
            nn.LeakyReLU(inplace=True),
            Resblock(64,64),
            Resblock(64,64),
            Resblock(64,64),
            Resblock(64,64),
            Resblock(64,64),
        )

    def forward(self,input,att):
        conv_1 = self.conv(torch.cat((input, att), 1))
        return conv_1

class Flash3(nn.Module):
    def __init__(self):
        super(Flash3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, f,input, att):
        f3 = self.conv(f)

        result = f3.mul(att)
        result = result + input
        return result,f3

