import numpy as np

import torch
import torch.nn as nn

# 通道注意力层(CALayer)
class CALayer(nn.Module):
    def __init__(self, channels=64, input_height=128, input_width=128, reduction=16):
        super(CALayer, self).__init__()

        # self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.global_average_pooling = nn.AvgPool2d(kernel_size=(input_height, input_width))

        self.conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, stride=1, padding=0)

        self.upsample_layer = nn.Upsample(size=(input_height, input_width), mode='bilinear', align_corners=False)
        
    def forward(self, x):
        gla_pooling = self.global_average_pooling(x)
        c1 = self.conv1(gla_pooling)
        #relu1 = self.leaky_relu(c1)
        relu1 = self.relu(c1)

        c2 = self.conv2(relu1)
        sig1 = self.sigmoid(c2)

        up = self.upsample_layer(sig1)
        output = x * up
        return output

# 通道注意力模块(CAB)
class RCAB(nn.Module):
    def __init__(self, channels=64, input_height=128, input_width=128):
        super(RCAB, self).__init__()

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.cal = CALayer(channels, input_height, input_width, reduction=16)

    def forward(self, x):
        c1 = self.conv1(x)
        lrelu1 = self.leaky_relu(c1)
        c2 = self.conv2(lrelu1)
        lrelu2 = self.leaky_relu(c2)
        att = self.cal(lrelu2)
        output = x + att
        return output
    
# 残差组(ResidualGroup)
class ResidualGroup(nn.Module):
    def __init__(self, n_RCAB=5, channels=64, input_height=128, input_width=128):
        super(ResidualGroup, self).__init__()

        CAB_list = [RCAB(channels, input_height, input_width) for i in range(n_RCAB)]
        self.CABs = nn.Sequential(*CAB_list)

    def forward(self, x):
        out = self.CABs(x) + x
        return out

# PFE，MPE特征融合后提取
class FCD(nn.Module):
    def __init__(self,input_channels=9, output_channels=64, input_height=128, input_width=128, n_rg=5, scale=2):
        super(FCD, self).__init__()

        self.conv1 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        ResGroup_list = [ResidualGroup(5, output_channels, input_height, input_width) for i in range(n_rg)]
        self.ResGroup = nn.Sequential(*ResGroup_list)

        self.conv2 = nn.Conv2d(output_channels, output_channels * (scale ** 2), kernel_size=3, stride=1, padding=1)
        self.cal = CALayer(output_channels * (scale ** 2), input_height, input_width)

        self.conv3 = nn.Conv2d(output_channels * (scale ** 2), input_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        
        return 

# 特征提取
class Feature_Extracte(nn.Module):
    def __init__(self, input_channel=9, output_channel=64, input_height=128, input_width=128, n_rg=5):
        super(Feature_Extracte, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        ResGroup_list = [ResidualGroup(5, output_channel, input_height, input_width) for i in range(n_rg)]
        self.ResGroup = nn.Sequential(*ResGroup_list)

        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        c1 = self.conv1(x)
        lrelu1 = self.leaky_relu(c1)
        rgs = self.ResGroup(lrelu1)
        c2 = self.conv2(rgs)
        lrelu2 = self.leaky_relu(c2)
        out = lrelu2
        return out

class rDL_Denoise(nn.Module):
    def __init__(self,input_channels=9, output_channels=64, input_height=128, input_width=128, n_rgs=[5, 2, 5]):
        super(rDL_Denoise, self).__init__()
        self.PFE = Feature_Extracte(input_channels, output_channels, input_height, input_width,n_rgs[0])
        self.MPE = Feature_Extracte(input_channels, output_channels, input_height, input_width,n_rgs[1])
        self.FCD = FCD(input_channels, output_channels, input_height, input_width,n_rgs[1])

    def forward(self, inputs_PFE, inputs_MPE):
        pfe = self.PFE(inputs_PFE)
        mpe = self.MPE(inputs_MPE)
        fcd = self.FCD(pfe + mpe)
        return fcd
    