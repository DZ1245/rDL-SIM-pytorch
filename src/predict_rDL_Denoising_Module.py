import os
import numpy as np
import tifffile as tiff
import torch
import cv2
import numpy.fft as F

import config.config_DN as config_DN
from utils.read_mrc import read_mrc
from utils.checkpoint import load_checkpoint
from utils.utils import prctile_norm

from sim_fitting.read_otf import read_otf
from sim_fitting.Parameters_2DSIM import parameters
from sim_fitting.CalModamp_2DSIM import cal_modamp

args, unparsed = config_DN.get_args()
cwd = os.getcwd()

root_path = args.root_path
data_folder = args.data_folder
DN_save_weights_path = args.DN_save_weights_path
SR_save_weights_path = args.SR_save_weights_path

SR_model_name = args.SR_model_name
DN_model_name = args.DN_model_name
SR_resume_name = args.SR_resume_name

input_height = args.input_height
input_width = args.input_width

load_weights_flag = args.load_weights_flag
DN_exp_name = args.exp_name
DN_resume_name = args.resume_name

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

DN_mode = 'test'
SR_mode = 'test'


# --------------------------------------------------------------------------------
#                                  GPU env set
# --------------------------------------------------------------------------------
local_rank = args.local_rank
torch.cuda.set_device(local_rank) 
device = torch.device("cuda", local_rank)

# define and make output dir
DN_save_weights_path = DN_save_weights_path + data_folder + "/"
SR_save_weights_path = SR_save_weights_path + data_folder + "/"

raw_path = os.path.join('../Demo/Raw/DN',data_folder)
result_path = os.path.join('../Demo/Result/DN',data_folder)
gt_path = os.path.join('../Demo/GT/DN',data_folder)

if not os.path.exists(result_path):
    os.makedirs(result_path)


# --------------------------------------------------------------------------------
#                               select models
# --------------------------------------------------------------------------------
if SR_model_name == "DFCAN":
    from model.DFCAN import DFCAN
    SR_model = DFCAN(n_ResGroup=4, n_RCAB=4, scale=2, input_channels=nphases*ndirs, mid_channels=64 , out_channels=1)
    print("SR:DFCAN model create")

if DN_model_name == "rDL_Denoiser":
    from model.rDL_Denoise import rDL_Denoise
    DN_model = rDL_Denoise(input_channels=nphases, output_channels=64, input_height=input_height, input_width=input_width)
    print("DN:rDL_Denoise model create")


# load model
assert load_weights_flag==1
_, _ = load_checkpoint(SR_save_weights_path, SR_resume_name, None, 
                       SR_mode, SR_model, None, None, local_rank)
_, _ = load_checkpoint(DN_save_weights_path, DN_resume_name, DN_exp_name, 
                        DN_mode, DN_model, None, None, local_rank)

SR_model.to(device)
DN_model.to(device)

# --------------------------------------------------------------------------------
#                         predefine OTF and other parameters
# --------------------------------------------------------------------------------
# define parameters
[Nx, Ny] = [pParam.Nx, pParam.Ny]
[dx, dy, dxy] = [pParam.dx, pParam.dy, pParam.dxy]
[dkx, dky, dkr] = [pParam.dkx, pParam.dky, pParam.dkr]
[nphases, ndirs] = [pParam.nphases, pParam.ndirs]
space = pParam.space
scale = pParam.scale
phase_space = 2 * np.pi / nphases

[Nx_hr, Ny_hr] = [Nx, Ny] * scale
[dx_hr, dy_hr] = [x / scale for x in [dx, dy]]

xx = dx_hr * np.arange(-Nx_hr / 2, Nx_hr / 2, 1)
yy = dy_hr * np.arange(-Ny_hr / 2, Ny_hr / 2, 1)
[X, Y] = np.meshgrid(xx, yy)

# read OTF and PSF
OTF, prol_OTF, PSF = read_otf(OTF_path_list[wave_length], Nx_hr, Ny_hr, dkx, dky, dkr)


# --------------------------------------------------------------------------------
#                         predict Demo
# --------------------------------------------------------------------------------
SR_model.eval()
DN_model.eval()
raw_list = os.listdir(raw_path)

for raw in raw_list:
    p = os.path.join(raw_path, raw)
    groud = os.path.join(gt_path, raw)
    print(p)

    if raw[-3:]=='mrc':
        header, data = read_mrc(p)
        height, width, channels = header['nx'][0], header['ny'][0], header['nz'][0]
        # input= data.astype(np.float32) #256 256 9
        # inputs = data.astype(np.float32).transpose(2, 1, 0)
        # inputs = np.flip(inputs, axis=1)
        # data_gt = torch.rand(1, 1, 256, 256)

    elif raw[-3:]=='tif':
        data = tiff.imread(p).astype(np.float32)
        # data_gt = tiff.imread(groud).astype(np.float32)
        channels, height, width = data.shape
        data = np.flip(data.transpose(2, 1, 0), axis=1)

    
    if height!=input_height or width!=input_width:
        print(p,height,width)
        continue

    # 这里默认num_average=1
    num_average = 1
    average_batch = np.zeros((Ny, Nx, ndirs * nphases))
    for j in range(num_average):
        average_batch = average_batch + data
    
    # 用途未知
    cur_k0, modamp = cal_modamp(np.array(average_batch), prol_OTF, pParam)
    cur_k0_angle = np.array(np.arctan(cur_k0[:, 1] / cur_k0[:, 0]))
    cur_k0_angle[1:3] = cur_k0_angle[1:3] + np.pi
    cur_k0_angle = -(cur_k0_angle - np.pi / 2)
    for nd in range(ndirs):
        if np.abs(cur_k0_angle[nd] - pParam.k0angle_g[nd]) > 0.05:
            cur_k0_angle[nd] = pParam.k0angle_g[nd]
    cur_k0 = np.sqrt(np.sum(np.square(cur_k0), 1))
    given_k0 = 1 / space
    cur_k0[np.abs(cur_k0 - given_k0) > 0.1] = given_k0

    # 归一化准备
    img_in = data.astype(np.float32).transpose(2, 1, 0)
    # img_in = img_in /65535.0
    img_in = prctile_norm(img_in)
    SR_img_in = torch.Tensor(img_in)

    # ------------------------------------------------------------------------------
    #                               SR model predict
    # ------------------------------------------------------------------------------
    SR_img_in = SR_img_in.unsqueeze(0).to(device)
    img_SR = prctile_norm(SR_model(SR_img_in).cpu().detach().numpy())
    # cv2.resize expects (H,W,C).
    img_SR = np.squeeze(img_SR)
    # img_SR.shape = 512 512
    img_SR = cv2.resize(img_SR, (Ny_hr, Nx_hr))

    # ------------------------------------------------------------------------------
    #                   intensity equalization for each orientation
    # ------------------------------------------------------------------------------
    img_in = np.array(img_in)
    mean_th_in = np.mean(img_in[:nphases, :, :])
    for d in range(1, ndirs):
        data_d = img_in[d * nphases:(d + 1) * nphases, :, :]
        img_in[d * nphases:(d + 1) * nphases, :, :] = data_d * mean_th_in / np.mean(data_d)
    
    # ------------------------------------------------------------------------------
    #                        generate pattern-modulated images
    # ------------------------------------------------------------------------------
    phase_list = - np.angle(modamp)
    img_gen = []
    for d in range(ndirs):
        alpha = cur_k0_angle[d]
        for i in range(nphases):
            kxL = cur_k0[d] * np.pi * np.cos(alpha)
            kyL = cur_k0[d] * np.pi * np.sin(alpha)
            kxR = -cur_k0[d] * np.pi * np.cos(alpha)
            kyR = -cur_k0[d] * np.pi * np.sin(alpha)
            phOffset = phase_list[d] + i * phase_space
            interBeam = np.exp(1j * (kxL * X + kyL * Y + phOffset)) + np.exp(1j * (kxR * X + kyR * Y))
            pattern = prctile_norm(np.square(np.abs(interBeam)))
            patterned_img_fft = F.fftshift(F.fft2(pattern * img_SR)) * OTF
            modulated_img = np.abs(F.ifft2(F.ifftshift(patterned_img_fft)))
            modulated_img = cv2.resize(modulated_img, (Ny, Nx))
            img_gen.append(modulated_img)
    img_gen = prctile_norm(np.array(img_gen))


    # ------------------------------------------------------------------------------
    #                           prepare rDL denoising module data
    # ------------------------------------------------------------------------------
    img_in = np.transpose(img_in, (1, 2, 0))
    img_gen = np.transpose(img_gen, (1, 2, 0))
    pred = []

    for d in range(ndirs):
        Gen = img_gen[:, :, d * ndirs:(d + 1) * nphases]
        input_img = img_in[:, :, d * ndirs:(d + 1) * nphases]
        input_MPE = np.reshape(Gen, (1, input_height, input_width, nphases))
        input_PFE = np.reshape(input_img, (1, input_height, input_width, nphases))
        input_MPE = torch.Tensor(input_MPE).permute(0, 3, 1, 2).to(device)
        input_PFE = torch.Tensor(input_PFE).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            outputs = DN_model(input_PFE, input_MPE)
        
        pr = outputs.detach().cpu().numpy()

        for pha in range(nphases):
            pred.append(np.squeeze(pr[:, pha, :, :]))
    pred = prctile_norm(np.array(pred))
    
    # ------------------------------------------------------------------------------
    #                           train rDL denoising module
    # ------------------------------------------------------------------------------

    out = np.uint16(pred * 65535)
    out = np.flip(out, axis=1)
    out_path = os.path.join(result_path,raw[:-4] + '_result.tif')
    print(out_path)
    tiff.imwrite(out_path, out)
