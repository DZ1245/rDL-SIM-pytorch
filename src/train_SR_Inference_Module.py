import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

import utils.config_SR as config_SR
from utils.loss import MSESSIMLoss, AverageMeter
from utils.pytorch_ssim import SSIM
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from utils.checkpoint import save_checkpoint

# --------------------------------------------------------------------------------
#                          instantiation for parameters
# --------------------------------------------------------------------------------
args, unparsed = config_SR.get_args()
cwd = os.getcwd()

root_path = args.root_path
data_folder = args.data_folder
save_weights_path = args.save_weights_path
save_weights_suffix = args.save_weights_suffix

load_weights_flag = args.load_weights_flag
model_name = args.model_name

num_gpu = args.num_gpu
gpu_id = args.gpu_id
mixed_precision = args.mixed_precision
total_epoch = args.total_epoch
sample_epoch = args.sample_epoch
validate_epoch = args.validate_epoch
validate_num = args.validate_num
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
beta1 = args.beta1
beta2 = args.beta2
ssim_weight = args.ssim_weight

dataset = args.dataset
exp_name = args.exp_name
input_height = args.input_height
input_width = args.input_width
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag
resize_flag = args.resize_flag
num_workers = args.num_workers
log_iter = args.log_iter
wf = 0

# define and make output dir
# 数据集位置
data_root = root_path + dataset

save_weights_path = save_weights_path + data_folder + save_weights_suffix + "/"
save_weights_file = save_weights_path + data_folder + "_SR"
exp_path = save_weights_path + exp_name + '/'

sample_path = exp_path  + "sampled/"
log_path = exp_path  + "log/"

if not os.path.exists(exp_path):
    os.makedirs(exp_path)
if not os.path.exists(save_weights_path):
    os.makedirs(save_weights_path)
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)


# --------------------------------------------------------------------------------
#                                  GPU env set
# --------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device('cuda' if args.cuda else 'cpu')
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

torch.manual_seed(args.random_seed)
if args.cuda:
    torch.cuda.manual_seed(args.random_seed)

# --------------------------------------------------------------------------------
#                        select models optimizer and loss
# --------------------------------------------------------------------------------
if model_name == "DFCAN":
    from model.DFCAN import DFCAN
    model = DFCAN(n_ResGroup=4, n_RCAB=4, scale=2, input_channels=input_channels, out_channels=64)
    print("DFCAN model create")
# Just make every model to DataParallel
# print(model)
model.double().to(device)
model = torch.nn.DataParallel(model)

optimizer = AdamW(model.parameters(), lr=start_lr, betas=(beta1,beta2))
# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=lr_decay_factor, patience=5, verbose=True)

# MSEloss + SSIMloss
loss_function = MSESSIMLoss(ssim_weight=ssim_weight)


# --------------------------------------------------------------------------------
#                         select dataset and dataloader
# --------------------------------------------------------------------------------
if dataset == 'Microtubules':
    from dataloader.Microtubules import get_loader

train_loader = get_loader('train', input_height, input_width, norm_flag, resize_flag, 
                          scale_factor, wf, batch_size, data_root,True,num_workers)
val_loader = get_loader('val', input_height, input_width, norm_flag, resize_flag, 
                        scale_factor, wf, batch_size, data_root,True,num_workers)

# 创建 GradScaler 以处理梯度缩放
scaler = GradScaler()

# --------------------------------------------------------------------------------
#                                   train model
# --------------------------------------------------------------------------------
def train(epoch):
    model.train()
    loss_function.train()
    Loss_av = AverageMeter()

    t = time.time()
    # 训练中使用autocast进行混合精度计算
    for batch_idx, batch_info in enumerate(train_loader):
        # 将模型和优化器放入自动混合精度上下文
        with autocast():
            inputs = batch_info['input'].to(device)
            gts = batch_info['gt'].to(device)
            # 前向传播
            outputs = model(inputs)
            loss = loss_function(outputs, gts)
        
        # 反向传播和梯度更新
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        Loss_av.update(loss.item())

        # 其他训练步骤...
        if  batch_idx!=0 and batch_idx % log_iter == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}\tLr: {:.6f}\tTime({:.2f})'.format(
                epoch, batch_idx, len(train_loader), Loss_av.avg, optimizer.param_groups[-1]['lr'], time.time() - t))
            t = time.time()
            Loss_av = AverageMeter()

        # # 测试代码
        # if(batch_idx > 5):
        #     break


# --------------------------------------------------------------------------------
#                                   Val model
# --------------------------------------------------------------------------------

def val(epoch):
    model.eval()
    loss_function.eval()
    Loss_av = AverageMeter()

    t = time.time()
    with torch.no_grad():
        # 训练中使用autocast进行混合精度计算
        for batch_idx, batch_info in enumerate(val_loader):
            with autocast():
                inputs = batch_info['input'].to(device)
                gts = batch_info['gt'].to(device)
                # 前向传播
                outputs = model(inputs)
                loss = loss_function(outputs, gts)
            Loss_av.update(loss.item())

            if  batch_idx!=0 and batch_idx % log_iter == 0:
                print('Val Epoch: {} [{}/{}]\tLoss: {:.6f}\tTime({:.2f})'.format(
                    epoch, batch_idx, len(val_loader), Loss_av.avg, time.time() - t))
                t = time.time()
                Loss_av = AverageMeter()
            
            # # 测试代码
            # if(batch_idx > 5):
            #     break
        

    return Loss_av.avg


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
    fig.savefig(sample_path + '%d.png' % epoch)
    plt.close()


# --------------------------------------------------------------------------------
#                                       Main
# --------------------------------------------------------------------------------
def main():
    min_loss = torch.finfo(torch.float32).max
    # 定义训练循环
    for epoch in range(total_epoch):
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
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'min_loss': min_loss
        }, is_best, args.exp_name, save_weights_path)

        # update optimizer policy
        scheduler.step(test_loss)
    

if __name__ == "__main__":
    main()