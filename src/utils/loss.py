import torch.nn as nn
from utils.pytorch_ssim import SSIM

class MSESSIMLoss(nn.Module):
    def __init__(self,mse_weight=1.0,ssim_weight=1e-2):
        super(MSESSIMLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

    def forward(self, predicted, target):
        # 均方误差函数
        mse_loss = nn.MSELoss()

        # 结构相似性函数
        ssim_loss = SSIM()

        # 计算损失
        mse_l = mse_loss(predicted, target)
        ssim_l = ssim_loss(predicted, target)

        # 计算组合损失
        combined_loss = self.mse_weight * mse_l + self.ssim_weight * ssim_l

        return combined_loss
