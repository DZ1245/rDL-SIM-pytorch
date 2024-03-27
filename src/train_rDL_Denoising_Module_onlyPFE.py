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

from sim_fitting.Parameters_2DSIM import parameters
from sim_fitting.CalModamp_2DSIM import cal_modamp

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
wave_length = args.wave_length
excNA = args.excNA
OTF_path_488 = args.OTF_path_488
OTF_path_560 = args.OTF_path_560
OTF_path_647 = args.OTF_path_647
OTF_path_list = {488: OTF_path_488, 560: OTF_path_560, 647: OTF_path_647}
pParam = parameters(input_height, input_width, wave_length * 1e-3, excNA, setup=0)

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
if DN_model_name == "rDL_Denoiser":
    from model.rDL_Denoise import rDL_Denoise
    DN_model = rDL_Denoise(input_channels=nphases, output_channels=64, 
                           input_height=input_height, input_width=input_width, 
                           attention_mode=DN_attention_mode,encoder_type=Encoder_type)
    print("DN:rDL_Denoise model create")

# optimizer
DN_optimizer = AdamW(DN_model.parameters(), lr=start_lr, betas=(beta1,beta2))
DN_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    DN_optimizer, mode='min', factor=lr_decay_factor, patience=4, verbose=True)

# load DN model
start_epoch = 0
min_loss = 1000.0
if load_weights_flag==1:
    start_epoch, min_loss = load_checkpoint(DN_save_weights_path, DN_resume_name, DN_exp_name, 
                                  DN_mode, DN_model, DN_optimizer, start_lr, local_rank)

# MSEloss + SSIMloss
loss_function = MSE_SSIMLoss(ssim_weight=ssim_weight)


DN_model.to(device)


# --------------------------------------------------------------------------------
#                         select dataset and dataloader
# --------------------------------------------------------------------------------
if dataset == 'Microtubules':
    from dataloader.Microtubules import get_loader_DN

# DN_dataloader数据未经过归一化处理
train_loader = get_loader_DN('train', batch_size, data_root, True, num_workers)
val_loader = get_loader_DN('val', batch_size, data_root, True, num_workers)


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
    # 有问题 和论文感觉不一样
    DN_model.train()
    loss_function.train()

    start_time = datetime.datetime.now()
    for batch_idx, batch_info in enumerate(train_loader):
        inputs = batch_info['input']
        gts = batch_info['gt']
        
        assert inputs.shape[0]==1 and gts.shape[0]==1
        # 9 128 128
        img_in = np.squeeze(inputs)
        img_gt = np.squeeze(gts)

        img_in = prctile_norm(img_in)
        img_gt = prctile_norm(img_gt)
        

        # ------------------------------------------------------------------------------
        #                   intensity equalization for each orientation
        # ------------------------------------------------------------------------------
        img_in = np.array(img_in)
        img_gt = np.array(img_gt)

        mean_th_in = np.mean(img_in[:nphases, :, :])
        for d in range(1, ndirs):
            data_d = img_in[d * nphases:(d + 1) * nphases, :, :]
            img_in[d * nphases:(d + 1) * nphases, :, :] = data_d * mean_th_in / np.mean(data_d)
        
        mean_th_gt = np.mean(img_gt[:nphases, :, :])
        for d in range(ndirs):
            data_d = img_gt[d * nphases:(d + 1) * nphases, :, :]
            img_gt[d * nphases:(d + 1) * nphases, :, :] = data_d * mean_th_gt / np.mean(data_d)
 
        # ------------------------------------------------------------------------------
        #                           prepare rDL denoising module data
        # ------------------------------------------------------------------------------
        img_in = np.transpose(img_in, (1, 2, 0))
        img_gt = np.transpose(img_gt, (1, 2, 0))

        input_PFE_batch = []
        gt_batch = []
        for i in range(ndirs):

            input_PFE_batch.append(img_in[:, :, i * nphases:(i + 1) * nphases])
            gt_batch.append(img_gt[:, :, i * nphases:(i + 1) * nphases])
        # shape 3 128 128 3

        input_PFE_batch = torch.Tensor(np.array(input_PFE_batch)).permute(0,3,1,2).to(device)
        gt_batch = torch.Tensor(np.array(gt_batch)).permute(0,3,1,2).to(device)

        # ------------------------------------------------------------------------------
        #                           train rDL denoising module
        # ------------------------------------------------------------------------------
        outputs = DN_model(input_PFE_batch, None)
        loss = loss_function(outputs, gt_batch)

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

        # # 测试代码
        # if(batch_idx > 201):
        #     break

# --------------------------------------------------------------------------------
#                                   Val model
# --------------------------------------------------------------------------------
def val(epoch):
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
            inputs = batch_info['input']
            gts = batch_info['gt']
            
            assert inputs.shape[0]==1 and gts.shape[0]==1
            # 9 128 128
            img_in = np.squeeze(inputs)
            img_gt = np.squeeze(gts)
            

            # ------------------------------------------------------------------------------
            #                   intensity equalization for each orientation
            # ------------------------------------------------------------------------------
            img_in = np.array(img_in)
            img_gt = np.array(img_gt)
            mean_th_in = np.mean(img_in[:nphases, :, :])
            for d in range(1, ndirs):
                data_d = img_in[d * nphases:(d + 1) * nphases, :, :]
                img_in[d * nphases:(d + 1) * nphases, :, :] = data_d * mean_th_in / np.mean(data_d)
            mean_th_gt = np.mean(img_gt[:nphases, :, :])
            for d in range(ndirs):
                data_d = img_gt[d * nphases:(d + 1) * nphases, :, :]
                img_gt[d * nphases:(d + 1) * nphases, :, :] = data_d * mean_th_gt / np.mean(data_d)


            # ------------------------------------------------------------------------------
            #                           prepare rDL denoising module data
            # ------------------------------------------------------------------------------
            img_in = np.transpose(img_in, (1, 2, 0))
            img_gt = np.transpose(img_gt, (1, 2, 0))

            input_PFE_batch = []
            gt_batch = []
            for i in range(ndirs):
                input_PFE_batch.append(img_in[:, :, i * nphases:(i + 1) * nphases])
                gt_batch.append(img_gt[:, :, i * nphases:(i + 1) * nphases])
            # shape 3 128 128 3

            input_PFE_batch = torch.Tensor(np.array(input_PFE_batch)).permute(0,3,1,2).to(device)
            gt_batch = torch.Tensor(np.array(gt_batch)).permute(0,3,1,2).to(device)

            # ------------------------------------------------------------------------------
            #                           train rDL denoising module
            # ------------------------------------------------------------------------------
            outputs = DN_model(input_PFE_batch, None)
            loss = loss_function(outputs, gt_batch)

            Loss_all.update(loss.item())

            mse_avg.update(mse_loss(outputs, gt_batch))

            out_tmp = outputs.detach().cpu().numpy()
            out_tmp = np.reshape(out_tmp, [nphases*ndirs, input_height, input_width])
            out_tmp = np.transpose(out_tmp,(1, 2, 0))
            gt_tmp = gt_batch.detach().cpu().numpy()
            gt_tmp = np.reshape(gt_tmp, [nphases*ndirs, input_height, input_width])
            gt_tmp = np.transpose(gt_tmp,(1, 2, 0))

            # ssim_avg.update(ssim(outputs, gt_batch)) # 3 3 128 128 
            ssim_avg.update(compare_ssim(gt_tmp, out_tmp, multichannel=True))
            psnr_avg.update(compare_psnr(gt_tmp, out_tmp, data_range=1))


            elapsed_time = datetime.datetime.now() - start_time
            # 其他训练步骤...
            if local_rank==0 and batch_idx!=0 and batch_idx % log_iter == 0:
                logging.info('Val Epoch: {} [{}/{}]\tLoss: {:.4f}\tSSIM: {:.4f} \t MSE: {:.4f}\tPSNR: {:.4f}\tTime({})'
                         .format(epoch, batch_idx, len(val_loader), Loss_all.avg, ssim_avg.avg, 
                                mse_avg.avg, psnr_avg.avg, elapsed_time)
                                 )
                start_time = datetime.datetime.now()
                # ssim_avg.reset()
                # mse_avg.reset()

            # # 测试代码
            # if(batch_idx > 201):
            #     break
    
    return Loss_all.avg

# --------------------------------------------------------------------------------
#                                  Sample images
# --------------------------------------------------------------------------------
def sample_img(epoch):
    DN_model.eval()

    mses, nrmses, psnrs, ssims = [], [], [], []
    input_show, pred_show, gt_show = [], [], []
    img_name = []
    with torch.no_grad():
        data_iterator = iter(val_loader)
        for i in range(3):
            val_batch = next(data_iterator)

            inputs = val_batch['input']
            gts = val_batch['gt']
            inputs_path = val_batch['imgpaths_input']
            img_name.append(inputs_path[0].split('/')[-1])
            
            assert inputs.shape[0]==1 and gts.shape[0]==1
            # 9 128 128
            img_in = np.squeeze(inputs)
            img_gt = np.squeeze(gts)

            img_in = prctile_norm(img_in)
            img_gt = prctile_norm(img_gt)
            
            
            # ------------------------------------------------------------------------------
            #                   intensity equalization for each orientation
            # ------------------------------------------------------------------------------
            img_in = np.array(img_in)
            img_gt = np.array(img_gt)
            mean_th_in = np.mean(img_in[:nphases, :, :])
            for d in range(1, ndirs):
                data_d = img_in[d * nphases:(d + 1) * nphases, :, :]
                img_in[d * nphases:(d + 1) * nphases, :, :] = data_d * mean_th_in / np.mean(data_d)
            mean_th_gt = np.mean(img_gt[:nphases, :, :])
            for d in range(ndirs):
                data_d = img_gt[d * nphases:(d + 1) * nphases, :, :]
                img_gt[d * nphases:(d + 1) * nphases, :, :] = data_d * mean_th_gt / np.mean(data_d)
            
            # ------------------------------------------------------------------------------
            #                           prepare rDL denoising module data
            # ------------------------------------------------------------------------------
            img_in = np.transpose(img_in, (1, 2, 0))
            img_gt = np.transpose(img_gt, (1, 2, 0))

            input_PFE_batch = []
            gt_batch = []
            for i in range(1):# 只取三个通道进行计算
                input_PFE_batch.append(img_in[:, :, i * nphases:(i + 1) * nphases])
                gt_batch.append(img_gt[:, :, i * nphases:(i + 1) * nphases])
            # shape 1 128 128 3

            input_PFE_batch = torch.Tensor(np.array(input_PFE_batch)).permute(0,3,1,2).to(device)
            gt_batch = torch.Tensor(np.array(gt_batch)).permute(0,3,1,2).to(device)

            # ------------------------------------------------------------------------------
            #                           train rDL denoising module
            # ------------------------------------------------------------------------------
            outputs = DN_model(input_PFE_batch, None)

            # 1 3 128 128 -> 3 128 128
            img_pred = np.squeeze(outputs.detach().cpu().numpy())
            img_gt = np.squeeze(gt_batch.detach().cpu().numpy())
            # print(img_gt.shape)
            mses, nrmses, psnrs, ssims = cal_comp(np.squeeze(img_gt), 
                                                  np.squeeze(img_pred),
                                                    mses, nrmses, psnrs, 
                                                    ssims)
            
            input_PFE = np.squeeze(input_PFE_batch.detach().cpu().numpy())
            input_show.append(input_PFE[0, :, :])
            gt_show.append(img_gt[0, :, :])
            pred_show.append(img_pred[0, :, :])

    r, c = 3, 3
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for row in range(r):
        # axs[row, 1].set_title(
        #     ' PSNR=%.4f, SSIM=%.4f' % (psnrs[row * nphases], ssims[row * nphases]))
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

        # # update optimizer policy
        DN_scheduler.step(test_loss)

if __name__ == "__main__":
    main()