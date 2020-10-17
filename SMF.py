###  
###  including smf block
###  
from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, scale_fac=2):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_fac),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

## SMF block implementation
## whether to add attention mechanism to enhance the performance
## un finished
## implementation-version1
class SMF_block(nn.Module):

    ## f_channels  from the least channels to the most channels
    def __init__(self, g_channel, f_channels, f_int):
        ##
        super(SMF_block, self).__init__()
        self.W_xArr = nn.ModuleList()
        self.psiArr = nn.ModuleList()
        self.relu = nn.ReLU(inplace=False)
        for i in range(len(f_channels)):
            self.W_xArr.append(nn.Sequential(
                nn.Conv2d(f_channels[i], f_int, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(f_int)
                ))
            self.psiArr.append(nn.Sequential(
                nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
                ))
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channel, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

    def forward(self, g, xs):
    ## the resolution of xs are downsampled, its resolution is the same as g
        assert len(xs) == len(self.W_xArr)
        ## forward inference
        g1 = self.W_g(g)
        out = []
        for i in range(len(xs)):
            x1 = self.W_xArr[i](xs[i])
            psi = self.relu(g1+x1)
            psi = self.psiArr[i](psi)
            out.append(xs[i]*psi)
        ## 未将out转换为tuple
        return out


## attention-dropout block



## SE-block implementation


###change DownSampleBlock and implement UpsampleBlock
# downsample block
# downsample ratio = (1/2)^d_rate
# the number of resolution decreases
# the number of channels increase
class DownsampleBlock(nn.Module):
    def __init__(self, channel_Num, d_rate = 1):
        super(DownsampleBlock, self).__init__()
        self.d_rate = d_rate
        self.downsample = nn.ModuleList()
        for i in range(d_rate):
            self.downsample.append(nn.Conv2d(channel_Num, channel_Num*2, kernel_size=3, 
                stride=2, padding=1))
            channel_Num = channel_Num*2
            self.downsample.append(nn.BatchNorm2d(channel_Num))
        self.relu = nn.ReLU(inplace=False)
    def forward(self, x):
        for _, tmodule in enumerate(self.downsample):
            x = tmodule(x)
        x = self.relu(x)
        return x

## 使用up_conv来做
## 使用只有最后一层有relu的up_conv来做
## 现在使用了up_conv来做
class UpsampleBlock(nn.Module):
    def __init__(self, channel_Num, u_rate=1):
        super(UpsampleBlock, self).__init__()
        self.u_rate = u_rate
        self.upsample = nn.ModuleList()
        for i in range(u_rate):
            ## use relu in intermedia layers
            self.upsample.append(up_conv(channel_Num, channel_Num//2))
            channel_Num = channel_Num//2
            #self.upsample.append(nn.BatchNorm2d(channel_Num))
    def forward(self, x):
        for _, tmodule in enumerate(self.upsample):
            x = tmodule(x)
        return x

## 需要看看能不能跑通
##conv2d stride=2  better than u_net_smf???
##n1=64 is the best parameter for some dataset? no?
class U_Net_smf(nn.Module):
    """
    SMFUNet implementation
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_smf, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

# 
#def __init__(self, g_channel, f_channels, f_int)
        self.Up4 = up_conv(filters[4], filters[3])
        #self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.SMF4 = SMF_block(g_channel=filters[3], f_channels=[filters[0]*8, filters[1]*4, 
        	filters[2]*2, filters[3]], f_int=filters[2])
        # dowmsample part
        self.down14 = DownsampleBlock( filters[0], 3)
        self.down24 = DownsampleBlock( filters[1], 2)
        self.down34 = DownsampleBlock( filters[2], 1)
        ## SE block+dropoutChannel?
        ## ADL?
        self.adjustChannels4 = nn.Conv2d(filters[0]*8+filters[1]*4+filters[2]*2+filters[3]*2, filters[3], 
        	kernel_size=1, stride=1, padding=0, bias=True)
        self.adjustChannelsBN4 = nn.BatchNorm2d(filters[3])
        self.Up_conv4 = conv_block(filters[3], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
#        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.SMF3 = SMF_block(g_channel=filters[2], f_channels=[filters[0]*4, filters[1]*2, 
        	filters[2]], f_int=filters[1])
        self.down13 = DownsampleBlock( filters[0], 2)
        self.down23 = DownsampleBlock( filters[1], 1)
        self.adjustChannels3 = nn.Conv2d(filters[0]*4+filters[1]*2+filters[2]*2, filters[2], 
        	kernel_size=1, stride=1, padding=0, bias=True)
        self.adjustChannelsBN3 = nn.BatchNorm2d(filters[2])
        self.Up_conv3 = conv_block(filters[2], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
#        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.SMF2 = SMF_block(g_channel=filters[1], f_channels=[filters[0]*2, filters[1]],
             f_int=filters[0])
        self.down12 = DownsampleBlock( filters[0], 1)
        self.adjustChannels2 = nn.Conv2d(filters[0]*2+filters[1]*2, filters[1], 
             kernel_size=1, stride=1, padding=0, bias=True)
        self.adjustChannelsBN2 = nn.BatchNorm2d(filters[1])
        self.Up_conv2 = conv_block(filters[1], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
#        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.SMF1 = SMF_block(g_channel=filters[0], f_channels=[filters[0]], 
        	 f_int=filters[0]//2)
        self.Up_conv1 = conv_block(filters[0]*2, filters[0])

        #### next writing the downsampling part and 
        ## change related channels

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d4 = self.Up4(e5)
        #print(d5.shape)
        #x4 = self.Att5(g=d5, x=e4)
        e14 = self.down14(e1)
        e24 = self.down24(e2)
        e34 = self.down34(e3)
        d4t = self.SMF4(g=d4, xs=[e14, e24, e34, e4])
        ##try to concatenate the list
        d4t.append(d4)
        d4t = torch.cat(d4t, dim=1)
        d4t = self.adjustChannels4(d4t)
        d4t = self.adjustChannelsBN4(d4t)
        #d4 = torch.cat(d4t, dim=1)
        d4 = self.Up_conv4(d4t)

        d3 = self.Up3(d4)
        #x3 = self.Att4(g=d4, x=e3)
        e13 = self.down13(e1)
        e23 = self.down23(e2)
        d3t = self.SMF3(g=d3, xs=[e13, e23, e3])
        ##try to concatenate the list
        d3t.append(d3)
        d3t = torch.cat(d3t, dim=1)
        d3t = self.adjustChannels3(d3t)
        d3t = self.adjustChannelsBN3(d3t)
        #d3 = torch.cat(d3t, dim=1)
        d3 = self.Up_conv3(d3t)

        d2 = self.Up2(d3)
        e12 = self.down12(e1)
        d2t = self.SMF2(g=d2, xs=[e12, e2])
        ##try to concatenate the list
        d2t.append(d2)
        d2t = torch.cat(d2t, dim=1)
        d2t = self.adjustChannels2(d2t)
        d2t = self.adjustChannelsBN2(d2t)
        #x2 = self.Att3(g=d3, x=e2)
        #d3 = torch.cat((x2, d3), dim=1)
        d2 = self.Up_conv2(d2t)

        d1 = self.Up1(d2)
        d1t = self.SMF1(g=d1, xs=[e1])
        #x1 = self.Att2(g=d2, x=e1)
        d1t.append(d1)
        d1t = torch.cat(d1t, dim=1)
        d1 = self.Up_conv1(d1t)

        out = self.Conv(d1)

      #  out = self.active(out)

        return out

## original unet上的conv等是否有一些可一个更改的呢?


##
## 尝试加入adl block
## UNet_smf_adl network
##  from .adl import ADL 
## 可以尝试只选用drop部分，还是使用全部
## 只在一个部分加入，还是每个部分都加入?
## opp - SE block
class U_Net_smf_adl(nn.Module):
    """
    SMFUNet implementation
    """
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_smf_adl, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

# 
#def __init__(self, g_channel, f_channels, f_int)
        self.Up4 = up_conv(filters[4], filters[3])
        #self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.SMF4 = SMF_block(g_channel=filters[3], f_channels=[filters[0], filters[1], 
        	filters[2], filters[3]], f_int=filters[2])
        # dowmsample part
        self.down14 = DownsampleBlock( filters[0], 3)
        self.down24 = DownsampleBlock( filters[1], 2)
        self.down34 = DownsampleBlock( filters[2], 1)
        ## SE block+dropoutChannel?
        ## ADL?
        self.adjustChannels4 = nn.Conv2d(filters[0]+filters[1]+filters[2]+filters[3], filters[3], 
        	kernel_size=1, stride=1, padding=0, bias=True)
        self.adjustChannelsBN4 = nn.BatchNorm2d(filters[3])
        self.Up_conv4 = conv_block(filters[3], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
#        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.SMF3 = SMF_block(g_channel=filters[2], f_channels=[filters[0], filters[1], 
        	filters[2]], f_int=filters[1])
        self.down13 = DownsampleBlock( filters[0], 2)
        self.down23 = DownsampleBlock( filters[1], 1)
        self.adjustChannels3 = nn.Conv2d(filters[0]+filters[1]+filters[2], filters[2], 
        	kernel_size=1, stride=1, padding=0, bias=True)
        self.adjustChannelsBN3 = nn.BatchNorm2d(filters[2])
        self.Up_conv3 = conv_block(filters[2], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
#        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.SMF2 = SMF_block(g_channel=filters[1], f_channels=[filters[0], filters[1]],
             f_int=filters[0])
        self.down12 = DownsampleBlock( filters[0], 1)
        self.adjustChannels2 = nn.Conv2d(filters[0]+filters[1], filters[1], 
             kernel_size=1, stride=1, padding=0, bias=True)
        self.adjustChannelsBN2 = nn.BatchNorm2d(filters[1])
        self.Up_conv2 = conv_block(filters[1], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
#        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.SMF1 = SMF_block(g_channel=filters[0], f_channels=[filters[0]], 
        	 f_int=filters[0]//2)
        self.Up_conv1 = conv_block(filters[0], filters[0])

        #### next writing the downsampling part and 
        ## change related channels

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d4 = self.Up4(e5)
        #print(d5.shape)
        #x4 = self.Att5(g=d5, x=e4)
        e14 = self.down14(e1)
        e24 = self.down24(e2)
        e34 = self.down34(e3)
        d4t = self.SMF4(g=d4, xs=[e14, e24, e34, e4])
        d4t = self.adjustChannels4(d4t)
        d4t = self.adjustChannelsBN4(d4t)
        #d4 = torch.cat(d4t, dim=1)
        d4 = self.Up_conv4(d4t)

        d3 = self.Up3(d4)
        #x3 = self.Att4(g=d4, x=e3)
        e13 = self.down13(e1)
        e23 = self.down23(e2)
        d3t = self.SMF3(g=d3, xs=[e13, e23, e3])
        d3t = self.adjustChannels3(d3t)
        d3t = self.adjustChannelsBN3(d3t)
        #d3 = torch.cat(d3t, dim=1)
        d3 = self.Up_conv3(d3t)

        d2 = self.Up2(d3)
        e12 = self.down12(e1)
        d2t = self.SMF2(g=d2, xs=[e12, e2])
        d2t = self.adjustChannels2(d2t)
        d2t = self.adjustChannelsBN2(d2t)
        #x2 = self.Att3(g=d3, x=e2)
        #d3 = torch.cat((x2, d3), dim=1)
        d2 = self.Up_conv2(d2t)

        d1 = self.Up2(d2)
        d1t = self.SMF1(g=d1, xs=e1)
        #x1 = self.Att2(g=d2, x=e1)
        #d2 = torch.cat((x1, d2), dim=1)
        d1 = self.Up_conv1(d1t)

        out = self.Conv(d1)

      #  out = self.active(out)

        return out


## UNet_smf_adl_msupervision network
## 最终版本的网络




### GFF module的两种实现方式
## implementation1 -- 在整个网络中定义Gl，
##当进行融合时再依据特征的相对大小进行采样和放缩
##按照gate 方式融合。
class U_Net_gffH(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(U_Net_gffH, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.gate1 = nn.Sequential(
            nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.Sigmoid())
        self.Conv2 = conv_block(filters[0], filters[1])
        self.gate2 = nn.Sequential(
            nn.Conv2d(filters[1], 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.Sigmoid())
        self.Conv3 = conv_block(filters[1], filters[2])
        self.gate3 = nn.Sequential(
            nn.Conv2d(filters[2], 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.Sigmoid())
        self.Conv4 = conv_block(filters[2], filters[3])
        self.gate4 = nn.Sequential(
            nn.Conv2d(filters[3], 1, kernel_size=1, stride=1, padding=0, bias=True),
            torch.nn.Sigmoid())
        self.Conv5 = conv_block(filters[3], filters[4])

        ##for downsampling features
        self.down14 = DownsampleBlock( filters[0], 3)
        self.down13 = DownsampleBlock( filters[0], 2)
        self.down12 = DownsampleBlock( filters[0], 1)
        self.down24 = DownsampleBlock( filters[1], 2)
        self.down23 = DownsampleBlock( filters[1], 1)
        self.up21 = UpsampleBlock( filters[1], 1)
        self.down34 = DownsampleBlock( filters[2], 1)
        self.up32 = UpsampleBlock(filters[2], 1)
        self.up31 = UpsampleBlock(filters[2], 2)
        self.up43 = UpsampleBlock(filters[3], 1)
        self.up42 = UpsampleBlock(filters[3], 2)
        self.up41 = UpsampleBlock(filters[3], 3)

        self.Up4 = up_conv(filters[4], filters[3])
        self.Up_conv4 = conv_block(filters[4], filters[3])

        self.Up3 = up_conv(filters[3], filters[2])
        self.Up_conv3 = conv_block(filters[3], filters[2])

        self.Up2 = up_conv(filters[2], filters[1])
        self.Up_conv2 = conv_block(filters[2], filters[1])

        self.Up1 = up_conv(filters[1], filters[0])
        self.Up_conv1 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

### 如果先进行上下采样再计算gate，那么将会比现在的
##先计算gate的方式更加复杂（现有做法）
    def forward(self, x):

        e1 = self.Conv1(x)
        g1 = self.gate1(e1)
        e1f = e1*g1
        e1f2 = self.down12(e1f)
        e1f3 = self.down13(e1f)
        e1f4 = self.down14(e1f)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        g2 = self.gate2(e2)
        e2f = e2*g2
        # e2f1 = F.interpolate( e2f, scale_factor=2, mode='bilinear')
        e2f1 = self.up21(e2f)
        e2f3 = self.down23(e2f)
        e2f4 = self.down24(e2f)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        g3 = self.gate3(e3)
        e3f = e3*g3
        # e3f1 = F.interpolate( e3f, scale_factor=4, mode='bilinear')
        # e3f2 = F.interpolate( e3f, scale_factor=2, mode='bilinear')
        e3f1 = self.up31(e3f)
        e3f2 = self.up32(e3f)
        e3f4 = self.down34(e3f)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        g4 = self.gate4(e4)
        e4f = e4*g4
        #e4f1 = F.interpolate( e3f, scale_factor=4, mode='bilinear')
        ## need to deal with channel number problem
        ##solu: use up_conv to change the dimensions of their channels
        e4f1 = self.up41(e4f)
        e4f2 = self.up42(e4f)
        e4f3 = self.up43(e4f)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d4 = self.Up4(e5)
        ##gff method
        ## new
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d2)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Up1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.Up_conv1(d1)

        out = self.Conv(d1)

        #d1 = self.active(out)

        return out


## implementation2 -- 先将各个特征图进行放缩操作
##之后再在放缩后的特征图上计算GL，
##按照gate 方式融合








