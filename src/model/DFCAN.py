import numpy as np

import torch
import torch.nn as nn

# CUDA存储数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 输入图像的二维截窗操作，并返回截窗后的图像
# 原理没理解透 但返回数值与tf相同
# img.shape = bs, ny, nx, ch || bs, ch, ny, nx
def apodize2d(img, napodize=10):
    # # img.shape需要被改变 改变channel维度位置
    # img = img.permute(0, 2, 3, 1)
    bs, ny, nx, ch = img.size()
    img_apo = img[:, napodize:ny-napodize, :, :]

    imageUp = img[:, 0:napodize, :, :]
    imageDown = img[:, ny-napodize:, :, :]
    diff = (imageDown.flip(1) - imageUp) / 2
    l = torch.arange(napodize)
    fact_raw = (1 - torch.sin((l + 0.5) / napodize * np.pi / 2)).to(device)
    fact = fact_raw.view(1, -1, 1, 1).to(torch.float32)
    fact = fact.expand(bs, -1, nx, ch)
    factor = diff * fact
    imageUp = imageUp + factor
    imageDown = imageDown - factor.flip(1)
    img_apo = torch.cat([imageUp, img_apo, imageDown], dim=1)

    imageLeft = img_apo[:, :, 0:napodize, :]
    imageRight = img_apo[:, :, nx-napodize:, :]
    img_apo = img_apo[:, :, napodize:nx-napodize, :]
    diff = (imageRight.flip(2) - imageLeft) / 2
    fact = fact_raw.view(1, 1, -1, 1).to(torch.float32)
    fact = fact.expand(bs, ny, -1, ch)
    factor = diff * fact
    imageLeft = imageLeft + factor
    imageRight = imageRight - factor.flip(2)
    img_apo = torch.cat([imageLeft, img_apo, imageRight], dim=2)

    # # 恢复channel维度位置
    # img_apo = img_apo.permute(0, 3, 1, 2)
    return img_apo

# 按照TF维度 bt ny nx ch
def fft2(input):
    input = apodize2d(input, napodize=10)
    temp = input.permute(0, 3, 1, 2)
    temp_complex = torch.complex(temp, torch.zeros_like(temp))
    fft = torch.fft.fftn(temp_complex, dim=(-2, -1))
    absfft = (torch.abs(fft) + 1e-8) ** 0.1

    output = absfft.permute(0, 2, 3, 1)
    return output

# 按照TF维度 bt ny nx ch
def fftshift(input):
    bs, h, w, ch = input.size()
    fs11 = input[:, -h // 2:h, -w // 2:w, :]
    fs12 = input[:, -h // 2:h, 0:w // 2, :]
    fs21 = input[:, 0:h // 2, -w // 2:w, :]
    fs22 = input[:, 0:h // 2, 0:w // 2, :]

    output = torch.cat([torch.cat([fs11, fs21], dim=1),
                        torch.cat([fs12, fs22], dim=1)], dim=2)
    
    # 调整大小为 (128, 128)
    output = output.permute(0, 3, 1, 2)
    output = torch.nn.functional.interpolate(output, size=(h, w),
                                             mode='bilinear', align_corners=False)
    output = output.permute(0, 2, 3, 1)
    
    return output

# 全局平均池化 按照PT维度 bt ch ny nx
def global_average_pooling(input):
    # 指定要计算平均值的维度
    output = torch.mean(input, dim=(2, 3), keepdim=True)
    return output

# 傅里叶通道注意力(FCALayer)
# 这里参照TF代码，需要对channel维度位置进行变换
class FCALayer(nn.Module):
    def __init__(self, mid_channels=64, reduction=16):
        super(FCALayer, self).__init__()

        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels // reduction, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mid_channels // reduction, mid_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 转为TF的为维度 bt ch ny nx -> bt ny nx ch
        x_tf = x.permute(0, 2, 3, 1)
        absfft1 = fft2(x_tf) # 1 128 128 64
        absfft1 = fftshift(absfft1)
        # 转为PT的为维度 bt ny nx ch-> bt ch ny nx
        absfft1_pt = absfft1.permute(0, 3, 1, 2)
        
        c1 = self.conv1(absfft1_pt)
        r1 = self.relu(c1)
        gla_pooling = global_average_pooling(r1)
        c2 = self.conv2(gla_pooling)
        r2 = self.relu(c2)
        c3 = self.conv3(r2)
        sig1 = self.sigmoid(c3)
        out = x * sig1
        
        return out

# 傅里叶通道注意力模块(FCAB)
class FCAB(nn.Module):
    def __init__(self, mid_channels=64):
        super(FCAB, self).__init__()

        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.fcat = FCALayer(mid_channels)

    def forward(self, x):
        c1 = self.conv1(x)
        g1 = self.gelu(c1)
        c2 = self.conv2(g1)
        g2 = self.gelu(c2)

        fcat = self.fcat(g2)
        output = x + fcat
        return output

# 残差组(ResidualGroup) 默认包含四个傅里叶通道注意力模块(FCAB)
class ResidualGroup(nn.Module):
    def __init__(self, n_RCAB=4, mid_channels=64):
        super(ResidualGroup, self).__init__()

        FCAB_list = [FCAB(mid_channels) for i in range(n_RCAB)]
        self.FCABs = nn.Sequential(*FCAB_list)

    def forward(self, x):
        out = self.FCABs(x) + x
        return out

# DFCAN模型
class DFCAN(nn.Module):
    def __init__(self, n_ResGroup=4, n_RCAB=4, scale=2, input_channels=9, mid_channels=64, out_channels=1):
        super(DFCAN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.gelu = nn.GELU()

        ResGroup_list = [ResidualGroup(n_RCAB, mid_channels) for i in range(n_ResGroup)]
        self.ResGroup = nn.Sequential(*ResGroup_list)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels * (scale ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape = bt * ch * nx * ny
        c1 = self.conv1(x) # bt, 64, 128, 128
        g1 = self.gelu(c1) # bt, 64, 128, 128
        
        resgroup = self.ResGroup(g1)
        
        c2 = self.conv2(resgroup)
        g2 = self.gelu(c2)

        # depth_to_space bt, 256, 128, 128 -> bt, 64, 256, 256
        pixshuff = self.pixel_shuffle(g2)
        c3 = self.conv3(pixshuff)
        out = self.sigmoid(c3)

        return out
