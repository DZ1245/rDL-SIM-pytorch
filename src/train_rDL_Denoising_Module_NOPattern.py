import os
import cv2
import logging
import datetime
import numpy as np
import numpy.fft as F
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

import config.config_DN as config_DN
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.loss import MSE_SSIMLoss, AverageMeter
from utils.utils import prctile_norm, cal_comp


# --------------------------------------------------------------------------------
#                          instantiation for parameters
# --------------------------------------------------------------------------------
args, unparsed = config_DN.get_args()
cwd = os.getcwd()

root_path = args.root_path
data_folder = args.data_folder
DN_save_weights_path = args.DN_save_weights_path
SR_save_weights_path = args.SR_save_weights_path

load_weights_flag = args.load_weights_flag
SR_model_name = args.SR_model_name
DN_model_name = args.DN_model_name
SR_resume_name = args.SR_resume_name
DN_attention_mode = args.DN_attention_mode
Encoder_type = args.Encoder_type

total_epoch = args.total_epoch
sample_epoch = args.sample_epoch

batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
beta1 = args.beta1
beta2 = args.beta2
ssim_weight = args.ssim_weight

dataset = args.dataset
DN_exp_name = args.exp_name
DN_resume_name = args.resume_name

input_height = args.input_height
input_width = args.input_width
scale_factor = args.scale_factor
norm_flag = args.norm_flag
resize_flag = args.resize_flag

num_workers = args.num_workers
log_iter = args.log_iter

# define SIM parameters
ndirs = args.ndirs
nphases = args.nphases

DN_mode = 'train'
SR_mode = 'test'

# define and make output dir
data_root = os.path.join(root_path, dataset)

DN_save_weights_path = os.path.join(DN_save_weights_path, data_folder)
DN_exp_path = os.path.join(DN_save_weights_path, DN_exp_name)
DN_sample_path = os.path.join(DN_exp_path, "sampled")
DN_log_path  = os.path.join(DN_exp_path, "log")

SR_save_weights_path  = os.path.join(SR_save_weights_path, data_folder)

if not os.path.exists(DN_save_weights_path):
    os.makedirs(DN_save_weights_path)
if not os.path.exists(DN_sample_path):
    os.makedirs(DN_sample_path)
if not os.path.exists(DN_log_path):
    os.makedirs(DN_log_path)


# --------------------------------------------------------------------------------
#                                  GPU env set
# --------------------------------------------------------------------------------
local_rank = args.local_rank
torch.cuda.set_device(local_rank) 
device = torch.device("cuda", local_rank)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)


# --------------------------------------------------------------------------------
#                        select models optimizer and loss
# --------------------------------------------------------------------------------
if SR_model_name == "DFCAN":
    from model.DFCAN import DFCAN
    SR_model = DFCAN(n_ResGroup=4, n_RCAB=4, scale=scale_factor, 
                     input_channels=nphases*ndirs, out_channels=1)
    print("SR:DFCAN model create")

if DN_model_name == "rDL_Denoiser":
    from model.rDL_Denoise_NoPattern import rDL_Denoise
    DN_model = rDL_Denoise(input_channels=nphases*ndirs, output_channels=64, 
                           input_height=input_height, input_width=input_width, 
                           attention_mode=DN_attention_mode,encoder_type=Encoder_type)
    print("DN:rDL_Denoise model create")

# optimizer
DN_optimizer = AdamW(DN_model.parameters(), lr=start_lr, betas=(beta1,beta2))
DN_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    DN_optimizer, mode='min', factor=lr_decay_factor, patience=4, verbose=True)

# load SR model
_, _ = load_checkpoint(SR_save_weights_path, SR_resume_name, None, 
                       SR_mode, SR_model, None, None, local_rank)

# load DN model
start_epoch = 0
min_loss = 1000.0
if load_weights_flag==1:
    start_epoch, min_loss = load_checkpoint(DN_save_weights_path, DN_resume_name, DN_exp_name, 
                                  DN_mode, DN_model, DN_optimizer, start_lr, local_rank)

# MSEloss + SSIMloss
loss_function = MSE_SSIMLoss(ssim_weight=ssim_weight)

SR_model.to(device)
DN_model.to(device)


# --------------------------------------------------------------------------------
#                         select dataset and dataloader
# --------------------------------------------------------------------------------
if dataset == 'Microtubules':
    from dataloader.Microtubules import get_loader_DN

# DN_dataloader数据未经过归一化处理
train_loader = get_loader_DN('train', batch_size, data_root, norm_flag, True, num_workers)
val_loader = get_loader_DN('val', batch_size, data_root, norm_flag, True, num_workers)


# --------------------------------------------------------------------------------
#                               define log writer
# --------------------------------------------------------------------------------
# logging
logging.basicConfig(level=logging.INFO)
log_file_path = os.path.join(DN_log_path, "log.txt")

# 创建一个文件处理器，用于写入日志到文件
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将文件处理器和流处理器添加到日志记录器
logging.getLogger().addHandler(file_handler)

if local_rank==0:
    logging.info(args)


# --------------------------------------------------------------------------------
#                                   train model
# --------------------------------------------------------------------------------
def train(epoch):
    SR_model.eval()
    DN_model.train()
    loss_function.train()

    start_time = datetime.datetime.now()
    for batch_idx, batch_info in enumerate(train_loader):
        inputs = batch_info['input'].to(device)
        gts = batch_info['gt'].to(device)

        img_SR = SR_model(inputs) # bt 1 256 256

        outputs = DN_model(inputs, img_SR)
        loss = loss_function(outputs, gts)

        # 不对SR进行训练
        DN_optimizer.zero_grad()
        loss.backward()
        DN_optimizer.step()

        elapsed_time = datetime.datetime.now() - start_time
        # 其他训练步骤...
        if local_rank==0 and batch_idx!=0 and batch_idx % log_iter == 0:
            logging.info('Train Epoch: {} [{}/{}]\tLoss: {:.8f}\tLr: {:.8f}\tTime({})'
                         .format(epoch, batch_idx, len(train_loader), loss.item(), 
                                 DN_optimizer.param_groups[-1]['lr'], elapsed_time)
                                 )
            start_time = datetime.datetime.now()

        # 测试代码
        # if(batch_idx > 10):
        #     break

# --------------------------------------------------------------------------------
#                                   Val model
# --------------------------------------------------------------------------------
def val(epoch):
    SR_model.eval()
    DN_model.eval()
    loss_function.eval()

    mse_loss = nn.MSELoss()
    # ssim = SSIM() 

    Loss_all = AverageMeter()
    mse_avg = AverageMeter()
    ssim_avg = AverageMeter()
    psnr_avg = AverageMeter()
    
    start_time = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, batch_info in enumerate(val_loader):
            inputs = batch_info['input'].to(device)
            gts = batch_info['gt'].to(device)
            
            img_SR = SR_model(inputs) # bt 1 256 256

            outputs = DN_model(inputs, img_SR)
            loss = loss_function(outputs, gts)
            
            outputs = DN_model(inputs, img_SR)
            loss = loss_function(outputs, gts)

            Loss_all.update(loss.item())
            mse_avg.update(mse_loss(outputs, gts))

            out_tmp = outputs.detach().cpu().numpy() # bt 9 128 128
            out_tmp = np.transpose(out_tmp,(0, 2, 3, 1))# bt 128 128 9
            gt_tmp = gts.detach().cpu().numpy()
            gt_tmp = np.transpose(gt_tmp,(0, 2, 3, 1))

            for i in range(batch_size):
                ssim_avg.update(compare_ssim(gt_tmp[i], out_tmp[i], multichannel=True))
                psnr_avg.update(compare_psnr(gt_tmp[i], out_tmp[i], data_range=1))

            elapsed_time = datetime.datetime.now() - start_time
            # 其他训练步骤...
            if local_rank==0 and batch_idx!=0 and batch_idx % log_iter == 0:
                logging.info('Val Epoch: {} [{}/{}]\tLoss: {:.4f}\tSSIM: {:.4f} \t MSE: {:.4f}\tPSNR: {:.4f}\tTime({})'
                         .format(epoch, batch_idx, len(val_loader), Loss_all.avg, ssim_avg.avg, 
                                mse_avg.avg, psnr_avg.avg, elapsed_time)
                                 )
                start_time = datetime.datetime.now()

            # # 测试代码
            # if(batch_idx > 10):
            #     break

    if local_rank==0:
        logging.info('Val Epoch: {} \tLoss: {:.4f}\tSSIM: {:.4f} \t MSE: {:.4f}\tPSNR: {:.4f}'
                    .format(epoch, Loss_all.avg, ssim_avg.avg, mse_avg.avg, psnr_avg.avg))
    
    return Loss_all.avg

# --------------------------------------------------------------------------------
#                                  Sample images
# --------------------------------------------------------------------------------
def sample_img(epoch):
    SR_model.eval()
    DN_model.eval()

    mses, nrmses, psnrs, ssims = [], [], [], []
    input_show, pred_show, gt_show = [], [], []
    img_name = []
    with torch.no_grad():
        data_iterator = iter(val_loader)
        val_batch = next(data_iterator)

        inputs = val_batch['input'][:3].to(device)
        gts = val_batch['gt'][:3].to(device)

        inputs_path = val_batch['imgpaths_input'][0:3]
        for i in range(3):
            img_name.append(inputs_path[i].split('/')[-1])
        
        
        img_SR = SR_model(inputs) # 3 1 256 256
        outputs = DN_model(inputs, img_SR) # 3 9 128 128


        # 1 3 128 128 -> 3 128 128
        img_pred = outputs.detach().cpu().numpy()
        img_gt = gts.detach().cpu().numpy()
        inputs_cpu = inputs.detach().cpu().numpy()
        # print(img_gt.shape)
        for i in range(3):
            mses, nrmses, psnrs, ssims = cal_comp(np.squeeze(img_gt[i]), np.squeeze(img_pred[i]), 
                                                  mses, nrmses, psnrs, ssims)
            input_show.append(inputs_cpu[i][0, :, :])
            gt_show.append(img_gt[i][0, :, :])
            pred_show.append(img_pred[i][0, :, :])

    r, c = 3, 3
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for row in range(r):
        # print(len(img_name),len(psnrs),len(ssims))
        axs[row, 1].set_title(
            'IMG=%s;PSNR=%.4f; SSIM=%4f' % (img_name[row],psnrs[row], ssims[row]))
        for col, image in enumerate([input_show, pred_show, gt_show]):
            axs[row, col].imshow(np.squeeze(image[row]))
            axs[row, col].axis('off')
        cnt += 1
        
    fig.savefig(os.path.join(DN_sample_path, '%d.png' % epoch))
    plt.close()


def main():
    # 定义训练循环
    global min_loss

    for epoch in range(start_epoch, total_epoch):
        train(epoch)
        # 模型保存和评估...
        test_loss = val(epoch)

        if epoch % sample_epoch == 0:
            sample_img(epoch)

        # save checkpoint
        is_best = test_loss < min_loss
        min_loss = min(test_loss, min_loss)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': DN_model.state_dict(), # 单GPU无module
            'optimizer': DN_optimizer.state_dict(),
            'min_loss': min_loss
        }, is_best, args.exp_name, DN_save_weights_path)

        # update optimizer policy
        DN_scheduler.step(test_loss)

if __name__ == "__main__":
    main()