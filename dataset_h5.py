import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import os
import cv2
import h5py
from tqdm import tqdm


import random # 导入随机库

class Dataset_Pro(data.Dataset):
    def __init__(self, file_path, img_scale, augment=False): # 增加 augment 参数
        super(Dataset_Pro, self).__init__()
        self.augment = augment # 是否开启增强

        with h5py.File(file_path, 'r') as data:
            print(f"loading Dataset_Pro: {file_path} with {img_scale}, Augment={augment}")
            # 读取数据并归一化
            self.gt = torch.from_numpy(data["gt"][...]).float() / img_scale
            self.ms = torch.from_numpy(data["ms"][...]).float() / img_scale
            self.lms = torch.from_numpy(data["lms"][...]).float() / img_scale
            self.pan = torch.from_numpy(data['pan'][...]).float() / img_scale

        print(f"Dataset loaded: PAN {self.pan.shape}")

    def __getitem__(self, index):
        gt = self.gt[index]
        lms = self.lms[index]
        ms = self.ms[index]
        pan = self.pan[index]

        # --- 数据增强逻辑 ---
        if self.augment:
            # 1. 随机水平翻转
            if random.random() > 0.5:
                gt = torch.flip(gt, dims=[2])
                lms = torch.flip(lms, dims=[2])
                ms = torch.flip(ms, dims=[2])
                pan = torch.flip(pan, dims=[2])

            # 2. 随机垂直翻转
            if random.random() > 0.5:
                gt = torch.flip(gt, dims=[1])
                lms = torch.flip(lms, dims=[1])
                ms = torch.flip(ms, dims=[1])
                pan = torch.flip(pan, dims=[1])

            # 3. 随机 90/180/270 度旋转
            if random.random() > 0.5:
                k = random.randint(1, 3) # 旋转次数
                gt = torch.rot90(gt, k, dims=[1, 2])
                lms = torch.rot90(lms, k, dims=[1, 2])
                ms = torch.rot90(ms, k, dims=[1, 2])
                pan = torch.rot90(pan, k, dims=[1, 2])

        return {'gt': gt, 'lms': lms, 'ms': ms, 'pan': pan}

    def __len__(self):
        return self.gt.shape[0]

def load_dataset_H5(file_path, scale):
    data = h5py.File(file_path)  # CxHxW
    print(data.keys())
    # tensor type:
    lms = torch.from_numpy(data['lms'][...] / scale).float()  # CxHxW = 8x64x64
    ms = torch.from_numpy(data['ms'][...] / scale).float()  # CxHxW= 8x64x64
    pan = torch.from_numpy(data['pan'][...] / scale).float()  # HxW = 256x256

    if data.get('gt', None) is None:
        gt = torch.from_numpy(data['lms'][...]).float()
    else:
        gt = torch.from_numpy(data['gt'][...]).float()

    return {'lms': lms,
            'ms': ms,
            'pan': pan,
            'gt': gt,}

class MultiExmTest_h5(data.Dataset):
    def __init__(self, file_path, img_scale):
        super(MultiExmTest_h5, self).__init__()
        self.img_scale = img_scale
        print(f"loading MultiExmTest_h5: {file_path} with {img_scale}")
        # 一次性载入到内存
        data = load_dataset_H5(file_path, img_scale)

        self.lms = data['lms']
        self.ms = data['ms']
        self.pan = data['pan']
        self.gt = data['gt']

        print(f"lms: {self.lms.shape}, ms: {self.ms.shape}, pan: {self.pan.shape}, gt: {self.gt.shape}")

    def __getitem__(self, item):
        return {'lms': self.lms[item, ...],
                'ms': self.ms[item, ...],
                'pan': self.pan[item, ...],
                'gt': self.gt[item, ...],}

    def __len__(self):
        return self.gt.shape[0]