import os
import numpy as np
import tifffile as tiff
import torch

import config.config_SR as config_SR
from utils.read_mrc import read_mrc
from utils.checkpoint import load_checkpoint
from utils.pytorch_ssim import SSIM
from utils.utils import prctile_norm


args, unparsed = config_SR.get_args()
cwd = os.getcwd()

root_path = args.root_path
data_folder = args.data_folder
save_weights_path = args.save_weights_path

load_weights_flag = args.load_weights_flag
model_name = args.model_name
norm_flag = args.norm_flag
exp_name = args.exp_name
resume_name = args.resume_name

input_channels = args.input_channels
out_channels = args.out_channels
scale_factor = args.scale_factor

mode = 'test'

local_rank = args.local_rank
torch.cuda.set_device(local_rank) 
device = torch.device("cuda", local_rank)

# define and make output dir
save_weights_path = os.path.join(save_weights_path, data_folder)

raw_path = os.path.join('../Demo/Raw/SR',data_folder)
result_path = os.path.join('../Demo/Result/SR',data_folder)
gt_path = os.path.join('../Demo/GT/SR',data_folder)

if not os.path.exists(result_path):
    os.makedirs(result_path)

if model_name == "DFCAN":
    from model.DFCAN import DFCAN
    model = DFCAN(n_ResGroup=4, n_RCAB=4, scale=scale_factor, input_channels=input_channels, mid_channels=64, out_channels=out_channels)
    print("DFCAN model create")
model.to(device)

assert load_weights_flag==1
_, min_loss = load_checkpoint(save_weights_path, resume_name, exp_name, mode, model, optimizer=None, lr=None, local_rank=local_rank)
print(min_loss)

model.eval()
raw_list = os.listdir(raw_path)
ssim = SSIM()

for raw in raw_list:
    p = os.path.join(raw_path, raw)
    groud = os.path.join(gt_path, raw)

    if raw[-3:]=='mrc':
        header, data = read_mrc(p)
        input_height, input_weight, channels = header['nx'][0], header['ny'][0], header['nz'][0]
        inputs = data.astype(np.float32).transpose(2, 1, 0)
        inputs = np.flip(inputs, axis=1)
        # data_gt = torch.rand(1, 1, 256, 256)

    elif raw[-3:]=='tif':
        data = tiff.imread(p).astype(np.float32)
        # data_gt = tiff.imread(groud).astype(np.float32)
        if len(data.shape) == 3:
            channels, input_height, input_weight = data.shape
        else:
            channels = 1
            input_height, input_weight = data.shape
        inputs = data

    if channels != input_channels:
        continue

    if norm_flag==1:
        inputs = prctile_norm(np.array(inputs))
        # gts = prctile_norm(np.array(data_gt))
        # print('prctile_norms')
    else:
        inputs = np.array(inputs) / 65535
        # gts = np.array(gts) / 65535

    inputs = torch.Tensor(inputs).unsqueeze(0).to(device)
    # gts = torch.Tensor(gts).unsqueeze(0).to(device)

    with torch.no_grad():
        if input_channels == 1:
            inputs = inputs.unsqueeze(1)
        outputs = model(inputs)

    # print(outputs.size())
    # print(ssim(outputs, gts))

    out = outputs[0].detach().cpu().numpy()
    out = np.uint16(out * 65535)
    out_path = os.path.join(result_path,raw[:-4] + '_result.tif')
    tiff.imwrite(out_path, out)




