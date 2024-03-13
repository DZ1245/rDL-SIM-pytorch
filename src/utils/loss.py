import torch.nn as nn
from utils.pytorch_ssim import SSIM

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MSE_SSIMLoss(nn.Module):
    def __init__(self,mse_weight=1.0,ssim_weight=1e-2):
        super(MSE_SSIMLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

    def forward(self, predicted, target):
        # 均方误差函数
        mse_loss = nn.MSELoss()

        # 结构相似性函数
        ssim_loss = SSIM()

        # 计算损失
        mse_l = mse_loss(predicted, target)
        ssim_l = 1 - ssim_loss(predicted, target)

        # 计算组合损失
        combined_loss = self.mse_weight * mse_l + self.ssim_weight * ssim_l

        return combined_loss
