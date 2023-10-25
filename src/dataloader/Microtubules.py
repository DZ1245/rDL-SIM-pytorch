import os
import glob
import numpy as np

from torch.utils.data import Dataset, DataLoader
from utils.utils import prctile_norm

import tifffile as tiff
import cv2

class Microtubules_SR(Dataset):
    def __init__(self, mode, height, width, norm_flag=1, resize_flag=0, scale=2, wf=0, data_root="/data/home/dz/rDL_SIM/SR/Microtubules"):
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
        self.norm_flag = norm_flag
        self.wf = wf

        print('[%d] images ready to be loaded' % len(self.imglist_input))


    def __getitem__(self, index):
        imgpaths_input = self.imglist_input[index]
        imgpaths_gt = self.imglist_gt[index]

        if imgpaths_input[-3:] == 'tif':
            # 直接单张tif
            curBatch = tiff.imread(imgpaths_input).astype(np.float32)
            gt = tiff.imread(imgpaths_gt).astype(np.float32)
        else:
            # 文件夹下存储图片数据
            img_path = glob.glob(imgpaths_input + '/*.tif')
            img_path.sort()
            curBatch = []
            for cur in img_path:
                img = tiff.imread(cur).astype(np.float32)
                if self.resize_flag == 1:
                    img = cv2.resize(img, (self.height * self.scale, self.width * self.scale))
                curBatch.append(img)
            #gt = imageio.imread(imgpaths_gt).astype(np.float)
            gt = tiff.imread(imgpaths_gt).astype(np.float32)
            
        # 增加归一化判断
        if self.norm_flag==1:
            curBatch = prctile_norm(np.array(curBatch))
            gt = prctile_norm(gt)
        else:
            curBatch = np.array(curBatch) / 65535
            gt = gt / 65535

        # wf进行降为判断 放在训练中算了
        # if self.wf == 1:
        #     image_batch = np.mean(image_batch, 3)
        #     for b in range(batch_size):
        #         image_batch[b, :, :] = prctile_norm(image_batch[b, :, :])
        #     image_batch = image_batch[:, :, :, np.newaxis]


        #print(type(curBatch), type(gt), type(imgpaths_input), type(imgpaths_gt))

        batch  = {
            'input' : curBatch,
            'gt' : gt,
            'imgpaths_input': imgpaths_input,
            'imgpaths_gt': imgpaths_gt
        }
        return batch

    def __len__(self):
        return len(self.imglist_input)
    
def get_loader_SR(mode, height, width, norm_flag, resize_flag, scale, wf, batch_size, data_root,shuffle=False, num_workers=0):
    dataset = Microtubules_SR(mode, height, width, norm_flag, resize_flag, scale, wf, data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

class Microtubules_DN(Dataset):
    def __init__(self, mode, data_root="/data/home/dz/rDL_SIM/SR/Microtubules_result"):
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

        print('[%d] images ready to be loaded' % len(self.imglist_input))


    def __getitem__(self, index):
        imgpaths_input = self.imglist_input[index]
        imgpaths_gt = self.imglist_gt[index]

        if imgpaths_input[-3:] == 'tif':
            # 直接单张tif
            curBatch = tiff.imread(imgpaths_input).astype(np.float32)
            gt = tiff.imread(imgpaths_gt).astype(np.float32)
        else:
            # 文件夹下存储图片数据
            img_path = glob.glob(imgpaths_input + '/*.tif')
            img_path.sort()
            curBatch = []
            for cur in img_path:
                img = tiff.imread(cur).astype(np.float32)
                curBatch.append(img)
            gt = tiff.imread(imgpaths_gt).astype(np.float32)
            
        batch  = {
            'input' : curBatch,
            'gt' : gt,
            'imgpaths_input': imgpaths_input,
            'imgpaths_gt': imgpaths_gt
        }
        return batch

    def __len__(self):
        return len(self.imglist_input)
    
def get_loader_DN(mode, batch_size, data_root,shuffle=False, num_workers=0):
    dataset = Microtubules_DN(mode, data_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)