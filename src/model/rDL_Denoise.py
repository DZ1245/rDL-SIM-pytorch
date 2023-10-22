import numpy as np

import torch
import torch.nn as nn

# 全局平均池化 按照PT维度 bt ch ny nx
def global_average_pooling(input):
    # 指定要计算平均值的维度
    output = torch.mean(input, dim=(2, 3), keepdim=True)
    return output

# 通道注意力层(CALayer)
class CALayer(nn.Module):
    def __init__(self, channels=64):
        super(CALayer, self).__init__()

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        # 上采样缺失
        
    def forward(self, x):
        gla_pooling = global_average_pooling(x)
        c1 = self.conv1(gla_pooling)
        lrelu1 = self.leaky_relu(c1)
        c2 = self.conv2(lrelu1)
        sig1 = self.sigmoid(c2)
        output = x * sig1
        return output

# 通道注意力模块(CAB)
class RCAB(nn.Module):
    def __init__(self, channels=64):
        super(RCAB, self).__init__()

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.cal = CALayer(channels)

    def forward(self, x):
        c1 = self.conv1(x)
        lrelu1 = self.leaky_relu(c1)
        c2 = self.conv1(lrelu1)
        lrelu2 = self.leaky_relu(c2)
        cal = self.cal(lrelu2)
        output = x + cal
        return output

class ResidualGroup(nn.Module):
    def __init__(self, n_RCAB=5, channels=64):
        super(ResidualGroup, self).__init__()

        CAB_list = [RCAB(channels) for i in range(n_RCAB)]
        self.CABs = nn.Sequential(*CAB_list)

    def forward(self, x):
        out = self.CABs(x) + x
        return out

# PFE，MPE特征融合后提取
class FCD(nn.Module):
    def __init__(self,input_channel, output_channel, n_rg=5):
        super(FCD, self).__init__()

    def forward(self, x):
        
        return 

# 特征提取
class Feature_Extracte(nn.Module):
    def __init__(self, input_channel=64, output_channel=64, n_rg=5):
        super(Feature_Extracte, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        ResGroup_list = [ResidualGroup(5, output_channel) for i in range(n_rg)]
        self.ResGroup = nn.Sequential(*ResGroup_list)

        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        
        return 

class rDL_Denoise(nn.Module):
    def __init__(self,input_channel, output_channel, n_rg=[5, 2, 5]):
        super(rDL_Denoise, self).__init__()

    def forward(self, inputs_PFE, inputs_MPE):
        
        return 
    
