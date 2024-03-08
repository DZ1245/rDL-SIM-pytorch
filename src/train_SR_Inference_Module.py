import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# from torch.utils.tensorboard import SummaryWriter   
from torch.optim import AdamW
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import config.config_SR as config_SR
from utils.loss import MSESSIMLoss, AverageMeter
from utils.pytorch_ssim import SSIM
from utils.checkpoint import save_checkpoint, load_checkpoint


# --------------------------------------------------------------------------------
#                          instantiation for parameters
# --------------------------------------------------------------------------------
args, unparsed = config_SR.get_args()
cwd = os.getcwd()

root_path = args.root_path
data_folder = args.data_folder
save_weights_path = args.save_weights_path

load_weights_flag = args.load_weights_flag
model_name = args.model_name

mixed_precision = args.mixed_precision
total_epoch = args.total_epoch
sample_epoch = args.sample_epoch
validate_epoch = args.validate_epoch
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
beta1 = args.beta1
beta2 = args.beta2
ssim_weight = args.ssim_weight

dataset = args.dataset
exp_name = args.exp_name
resume_name = args.resume_name
input_height = args.input_height
input_width = args.input_width
input_channels = args.input_channels
out_channels = args.out_channels

scale_factor = args.scale_factor
norm_flag = args.norm_flag
resize_flag = args.resize_flag
num_workers = args.num_workers
log_iter = args.log_iter
wf = 0
mode = 'train'

# time_now = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

# define and make output dir
# 数据集位置
data_root = os.path.join(root_path,dataset)
save_weights_path = os.path.join(save_weights_path, data_folder)
exp_path = os.path.join(save_weights_path, exp_name)
sample_path = os.path.join(exp_path,"sampled")
log_path = os.path.join(exp_path,"log")

if not os.path.exists(save_weights_path):
    os.makedirs(save_weights_path)
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

# --------------------------------------------------------------------------------
#                                  GPU env set
# --------------------------------------------------------------------------------

# DDP Setting
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl') 
device = torch.device("cuda", local_rank)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

# --------------------------------------------------------------------------------
#                        select models optimizer and loss
# --------------------------------------------------------------------------------
if model_name == "DFCAN" :
    from model.DFCAN import DFCAN
    model = DFCAN(n_ResGroup=4, n_RCAB=4, scale=scale_factor, input_channels=input_channels, mid_channels=64, out_channels=out_channels)
    print("DFCAN model create")

elif model_name == "DFCAN_SimAM" :
    from model.DFCAN_SimAM import DFCAN_SimAM
    model = DFCAN_SimAM(n_ResGroup=4, n_RCAB=4, scale=scale_factor, input_channels=input_channels, mid_channels=64, out_channels=out_channels)
    print("DFCAN_SimAM model create")

model.to(device)


optimizer = AdamW(model.parameters(), lr=start_lr, betas=(beta1,beta2))
# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=lr_decay_factor, patience=4, verbose=True, eps=1e-08)

# If resume, load checkpoint: model + optimizer
start_epoch = 0
min_loss = 1000.0
if load_weights_flag==1:
    start_epoch, min_loss = load_checkpoint(save_weights_path, resume_name, exp_name, mode, model, optimizer, start_lr, local_rank)

model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# MSEloss + SSIMloss
loss_function = MSESSIMLoss(ssim_weight=ssim_weight)

# --------------------------------------------------------------------------------
#                         select dataset and dataloader
# --------------------------------------------------------------------------------
if dataset == 'Microtubules':
    from dataloader.Microtubules import get_loader_SR

# SR_dataloader数据经过归一化处理
train_loader = get_loader_SR('train', input_height, input_width, norm_flag, resize_flag, 
                          scale_factor, wf, batch_size, data_root,True,num_workers)
val_loader = get_loader_SR('val', input_height, input_width, norm_flag, resize_flag, 
                        scale_factor, wf, batch_size, data_root,True,num_workers)


# --------------------------------------------------------------------------------
#                               define log writer
# --------------------------------------------------------------------------------
# writer = SummaryWriter(log_path)
# def write_log(writer, names, logs, epoch):
#     writer.add_scalar(names, logs, epoch)

# logging
logging.basicConfig(level=logging.INFO)
log_file_path = os.path.join(log_path, "log.txt")

# 创建一个文件处理器，用于写入日志到文件
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将文件处理器和流处理器添加到日志记录器
logging.getLogger().addHandler(file_handler)

logging.info(args)
# --------------------------------------------------------------------------------
#                                   train model
# --------------------------------------------------------------------------------
def train(epoch):
    model.train()
    loss_function.train()
    # Loss_av = AverageMeter()

    t = time.time()
    for batch_idx, batch_info in enumerate(train_loader):
        inputs = batch_info['input'].to(device)
        gts = batch_info['gt'].to(device)

        # bt h w -> bt c h w
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(1)
        if len(gts.shape) == 3:
            gts = gts.unsqueeze(1)

        # 前向传播
        outputs = model(inputs)

        loss = loss_function(outputs, gts)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss_av.update(loss.item())

        # 其他训练步骤...
        if  local_rank==0 and batch_idx!=0 and batch_idx % log_iter == 0:
            logging.info('Train Epoch: {} [{}/{}]\tLoss: {:.8f}\tLr: {:.8f}'
                         .format(epoch, batch_idx, len(train_loader), loss.item(), 
                                 optimizer.param_groups[-1]['lr'])
                                 )
        
            # print('Train Epoch: {} [{}/{}]\tLoss: {:.8f}\tLr: {:.8f}\tTime({:.2f})'.format(
            #     epoch, batch_idx, len(train_loader), Loss_av.avg, optimizer.param_groups[-1]['lr'], time.time() - t))
            # t = time.time()
            # Loss_av = AverageMeter()

        # 测试代码
        # if(batch_idx > 5):
        #     break


# --------------------------------------------------------------------------------
#                                   Val model
# --------------------------------------------------------------------------------
def val(epoch):
    model.eval()
    loss_function.eval()
    mse_loss = nn.MSELoss()
    ssim = SSIM() 

    # Loss_av = AverageMeter()
    mse_av = AverageMeter()
    ssim_av = AverageMeter()

    # t = time.time()
    with torch.no_grad():
        for batch_idx, batch_info in enumerate(val_loader):
            inputs = batch_info['input'].to(device)
            gts = batch_info['gt'].to(device)

            # bt h w -> bt c h w
            if len(inputs.shape) == 3:
                inputs = inputs.unsqueeze(1)
            if len(gts.shape) == 3:
                gts = gts.unsqueeze(1)

            # 前向传播
            outputs = model(inputs)

            loss = loss_function(outputs, gts)
            # Loss_av.update(loss.item())

            mse_av.update(mse_loss(outputs, gts))
            ssim_av.update(ssim(outputs, gts))
            
            # 测试代码
            # if(batch_idx > 5):
            #     break

    if local_rank==0:
        logging.info('Val Epoch: {}\tLoss: {:.8f} \tSSIM: {:.8f} \t MSE: {:.8f}'
                     .format(epoch, loss.item(), ssim_av.avg, mse_av.avg)
                     )
    # print('Val Epoch: {} \tLoss: {:.6f} \tSSIM: {:.6f} \t MSE: {:.6f} \tTime({:.2f})'.format(
    #         epoch, Loss_av.avg, ssim_av.avg, mse_av.avg, time.time() - t))
    
    # write_log(writer, 'Loss', Loss_av.avg, epoch)
    # write_log(writer, 'SSIM', ssim_av.avg, epoch)
    # write_log(writer, 'MSE', mse_av.avg, epoch)

    return loss.item()


# --------------------------------------------------------------------------------
#                                   Sample images
# --------------------------------------------------------------------------------
def sample_img(epoch):
    model.eval()
    mse_loss = nn.MSELoss()
    ssim = SSIM() 

    data_iterator = iter(val_loader)
    val_batch = next(data_iterator)

    inputs = val_batch['input'][:3].to(device)
    gts = val_batch['gt'][:3].to(device)

    # bt h w -> bt c h w
    if len(inputs.shape) == 3:
        inputs = inputs.unsqueeze(1)
    if len(gts.shape) == 3:
        gts = gts.unsqueeze(1)

    outputs = model(inputs)

    r, c = 3, 3
    img_show, gt_show, output_show = [], [], []
    mses, ssims, psnrs = [], [], []
    
    for  i in range(3):
        img_out = outputs[i].detach().cpu().numpy()
        img_gt = gts[i].detach().cpu().numpy()
        img_input = inputs[i].detach().cpu().numpy()

        img_show.append(np.mean(img_input,axis=0))
        gt_show.append(img_gt)
        output_show.append(img_out)
        mses.append(mse_loss(outputs[i], gts[i]))
        ssims.append(ssim(outputs[i].unsqueeze(0), gts[i].unsqueeze(0)))

        data_range = np.max(img_gt) - np.min(img_gt)
        psnrs.append(compare_psnr(img_gt, img_out, data_range=data_range))

    # show some examples
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for row in range(r):
        axs[row, 1].set_title('MSE=%.4f, SSIM=%.4f, PSNR=%.4f' % (mses[row], ssims[row], psnrs[row]))
        for col, image in enumerate([img_show, output_show, gt_show]):
            axs[row, col].imshow(np.squeeze(image[row]))
            axs[row, col].axis('off')
        cnt += 1 
    fig.savefig(os.path.join(sample_path,'%d.png' % epoch))
    plt.close()

# --------------------------------------------------------------------------------
#                                       Main
# --------------------------------------------------------------------------------
def main():
    global min_loss
    
    # 如果是继续训练，先看一下效果
    # if load_weights_flag==1:
    #     _ = val(start_epoch-1)

    # 定义训练循环
    for epoch in range(start_epoch, total_epoch):
        train(epoch)

        # 模型保存和评估
        test_loss = val(epoch)

        if epoch % sample_epoch == 0:
            sample_img(epoch)
        
        # save checkpoint
        is_best = test_loss < min_loss
        min_loss = min(test_loss, min_loss)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_loss': min_loss
        }, is_best, args.exp_name, save_weights_path)

        # update optimizer policy
        scheduler.step(test_loss)
        
    # 寻找没有使用的参数
    # train(1)
    # for name, param in model.named_parameters():
    #     if param.grad is None:
    #         print(name)

if __name__ == "__main__":
    main()