import os
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

import utils.config_SR as config_SR
from utils.loss import MSESSIMLoss

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
validate_interval = args.validate_interval
validate_num = args.validate_num
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
beta1 = args.beta1
beta2 = args.beta2
ssim_weight = args.ssim_weight

dataset = args.dataset
input_height = args.input_height
input_width = args.input_width
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag
resize_flag = args.resize_flag
num_workers = args.num_workers
wf = 0

# define and make output dir
# 数据集位置
data_root = root_path + data_folder
save_weights_path = save_weights_path + data_folder + save_weights_suffix + "/"
save_weights_file = save_weights_path + data_folder + "_SR"

sample_path = save_weights_path + "sampled/"
log_path = save_weights_path + "log/"
if not os.path.exists(save_weights_path):
    os.mkdir(save_weights_path)
if not os.path.exists(sample_path):
    os.mkdir(sample_path)
if not os.path.exists(log_path):
    os.mkdir(log_path)


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
# Just make every model to DataParallel
# print(model)
model = torch.nn.DataParallel(model).to(device)

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

train_loader = get_loader('train', input_height, input_width, norm_flag, resize_flag, scale_factor, wf, batch_size, data_root,True,num_workers)
val_loader = get_loader('val', input_height, input_width, norm_flag, resize_flag, scale_factor, wf, batch_size, data_root,True,num_workers)

# 创建 GradScaler 以处理梯度缩放
scaler = GradScaler()

# 训练中使用autocast进行混合精度计算
# 定义训练循环
for epoch in range(total_epoch):
    for batch_idx, batch_info in enumerate(train_loader):
        break
        # 将模型和优化器放入自动混合精度上下文
        # with autocast():
            # 前向传播
        #     outputs = model(inputs)
        #     loss = compute_loss(outputs, targets)
        
        # # 反向传播和梯度更新
        # optimizer.zero_grad()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        
        # 其他训练步骤...
    break
# 模型保存和评估...