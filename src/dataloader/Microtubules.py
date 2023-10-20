import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import tifffile as tiff
import imageio
import cv2

# 归一化尚未完成

class Microtubules_SR(Dataset):
    def __init__(self, mode, height, width, resize_flag=0, scale=2, data_root="/data/home/dz/rDL_SIM/SR/Microtubules_result"):
        # 根据模式选择数据
        if mode == "train":
            input_path = os.path.join(data_root,'train')
            input_name = os.listdir(input_path)
            gt_path = os.path.join(data_root,'train_gt')
            gt_name = os.listdir(gt_path)

        elif mode == "val":
            input_path = os.path.join(data_root,'val')
            input_name = os.listdir(input_path)
            gt_path = os.path.join(data_root,'val_gt')
            gt_name = os.listdir(gt_path)
        
        assert len(gt_name) == len(input_name)

        self.imglist_input = []
        for name in input_name:
            self.imglist_input.append(os.path.join(input_path, name))

        self.imglist_gt = []
        for name in input_name:
            self.imglist_gt.append(os.path.join(gt_path, name))

        self.resize_flag = resize_flag
        self.scale = scale
        self.height = height
        self.width = width

        print('[%d] images ready to be loaded' % len(self.imglist_input))


    def __getitem__(self, index):
        imgpaths_input = self.imglist_input[index]
        imgpaths_gt = self.imglist_gt[index]

        if imgpaths_input[-3:] == 'tif':
            # 直接单张tif
            curBatch = tiff.imread(imgpaths_input).astype(np.float)
            gt = imageio.imread(imgpaths_gt).astype(np.float)
        else:
            # 文件夹下存储图片数据
            img_path = glob.glob(imgpaths_input + '/*.tif')
            img_path.sort()
            curBatch = []
            for cur in img_path:
                img = imageio.imread(cur).astype(np.float)
                if self.resize_flag == 1:
                    img = cv2.resize(img, (self.height * self.scale, self.width * self.scale))
                curBatch.append(img)
            gt = imageio.imread(imgpaths_gt).astype(np.float)

        # 增加归一化判断

        batch  = {
            'input' : curBatch,
            'gt' : gt,
            'imgpaths_input': imgpaths_input,
            'imgpaths_gt': imgpaths_gt
        }
        return batch

    def __len__(self):
        return len(self.imglist_input)
    
def get_loader(mode, height, width, resize_flag, scale, batch_size, data_root,shuffle=False, num_workers=0):
    dataset = Microtubules_SR(mode, height, width, resize_flag, scale, data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)