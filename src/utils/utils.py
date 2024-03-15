import os
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# 百分比归一化
def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    return y

def cal_comp(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None):
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    gt, pr = np.squeeze(gt), np.squeeze(pr)
    gt = gt.astype(np.float32)

    # print(gt.shape,pr.shape) 3 128 128
    if gt.ndim == 2:
        n = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        n = np.size(gt, 0)

    # for i in range(n):
    #     mses.append(compare_mse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
    #     nrmses.append(compare_nrmse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
    #     psnrs.append(compare_psnr(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i])), data_range=1))
    #     ssims.append(compare_ssim(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
    
    # 所有通道一起计算
    gt = np.transpose(gt, (1, 2, 0))
    pr = np.transpose(pr, (1, 2, 0)) # H W C
    mses.append(compare_mse(prctile_norm(np.squeeze(gt)), prctile_norm(np.squeeze(pr))))
    nrmses.append(compare_nrmse(prctile_norm(np.squeeze(gt)), prctile_norm(np.squeeze(pr))))
    psnrs.append(compare_psnr(prctile_norm(np.squeeze(gt)), prctile_norm(np.squeeze(pr)), data_range=1))
    ssims.append(compare_ssim(prctile_norm(np.squeeze(gt)), prctile_norm(np.squeeze(pr)), multichannel=True))
    
    return mses, nrmses, psnrs, ssims