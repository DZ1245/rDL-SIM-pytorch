import os
import numpy as np
import tifffile as tiff
import torch
from torch.cuda.amp import autocast

import utils.config_SR as config_SR
from utils.read_mrc import read_mrc
from utils.checkpoint import load_checkpoint

def prctile_norm(x, min_prc=0, max_prc=100):
    y = (x-np.percentile(x, min_prc))/(np.percentile(x, max_prc)-np.percentile(x, min_prc)+1e-7)
    return y

args, unparsed = config_SR.get_args()
cwd = os.getcwd()

root_path = args.root_path
data_folder = args.data_folder
save_weights_path = args.save_weights_path
save_weights_suffix = args.save_weights_suffix

load_weights_flag = args.load_weights_flag
model_name = args.model_name
norm_flag = args.norm_flag
exp_name = args.exp_name
resume_name = args.resume_name

mode = 'test'

device = torch.device('cuda' if args.cuda else 'cpu')

# define and make output dir
save_weights_path = save_weights_path + data_folder + save_weights_suffix + "/"

raw_path = os.path.join('./Demo/Raw',data_folder)
result_path = os.path.join('./Demo/Result',data_folder)

if not os.path.exists(result_path):
    os.makedirs(result_path)

if model_name == "DFCAN":
    from model.DFCAN import DFCAN
    model = DFCAN(n_ResGroup=4, n_RCAB=4, scale=2, input_channels=9, out_channels=64)
    print("DFCAN model create")
model.to(device)

assert load_weights_flag==1
_ = load_checkpoint(save_weights_path, resume_name, exp_name, mode, model, None, None)

model.eval()
raw_list = os.listdir(raw_path)
for raw in raw_list:
    p = os.path.join(raw_path, raw)
    if raw[-3:]=='mrc':
        header, data = read_mrc(p)
        input_height, input_weight, input_channels = header['nx'][0], header['ny'][0], header['nz'][0]
        inputs = data.astype(np.float32).transpose(2, 0, 1)
    elif raw[-3:]=='tif':
        data = tiff.imread(p)
        input_channels, input_height, input_weight = data.shape
        inputs = data.astype(np.float32)

    assert input_channels==9

    if norm_flag:
        inputs = prctile_norm(np.array(inputs))
    else:
        inputs = np.array(inputs) / 65535

    inputs = torch.Tensor(inputs).unsqueeze(0)

    print(inputs.size())
    with torch.no_grad():
        inputs = inputs.to(device)
        outputs = model(inputs)
    
    out = outputs[0].detach().cpu().numpy()
    out = out * 65535
    out_path = os.path.join(result_path,raw[:-3] + 'tif')
    print(outputs)
    tiff.imwrite(out_path, out)



