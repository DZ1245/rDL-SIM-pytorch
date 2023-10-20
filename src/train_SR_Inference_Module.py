import os
import torch
from torch.optim import AdamW

import utils.config_SR as config_SR

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
total_iterations = args.total_iterations
sample_interval = args.sample_interval
validate_interval = args.validate_interval
validate_num = args.validate_num
batch_size = args.batch_size
start_lr = args.start_lr
lr_decay_factor = args.lr_decay_factor
beta1 = args.beta1
beta2 = args.beta2

input_height = args.input_height
input_width = args.input_width
input_channels = args.input_channels
scale_factor = args.scale_factor
norm_flag = args.norm_flag

# define and make output dir
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
#                           select models and optimizer
# --------------------------------------------------------------------------------
if model_name == "DCFAN":
    from model.DFCAN import DFCAN
    model = DFCAN(n_ResGroup=4, n_RCAB=4, scale=2, input_channels=input_channels, out_channels=64)
# Just make every model to DataParallel
model = torch.nn.DataParallel(model).to(device)

optimizer = AdamW(model.parameters(), lr=start_lr, betas=(beta1,beta2))
# Learning Rate Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=lr_decay_factor, patience=5, verbose=True)

# 训练中使用autocast进行混合精度计算
