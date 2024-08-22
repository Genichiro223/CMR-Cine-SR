import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import glob
import einops

# class Cine(Dataset):
#     def __init__(self, path, transform=None):
#         super(Cine, self).__init__()
#         self.file_path = path 
#         self.file = np.load(path)  # .astype(np.float32)
#         self.transform = transform
#         hr = glob.glob(path + '/HR.npy')
#         lr = glob.glob(path + '/LR.npy')
        

#     def __getitem__(self, index): 
#         image = self.file[index]  # 2, 128, 128
#         image = image.transpose((1, 2, 0))  # 128, 128, 2
#         # target = image[...,0]
#         # source = image[...,1]
#         # if self.transform:
#         #     target = self.transform(target)
#         #     source = self.transform(source)
#         if self.transform:
#             image = self.transform(image)
#         target = image[0]
#         source = image[1]
#         return target, source

#     def __len__(self):
#         return self.file.shape[0]

class Cine(Dataset):
    def __init__(self, path, transform=None):
        super(Cine, self).__init__()
        self.file_path = path 
        self.file = np.load(path)  # 680, 2, 24, 128, 128
        self.transform = transform
        

    def __getitem__(self, index): 
        image = self.file[index]  # 2, 24, 128, 128
        if len(image.shape) == 4:  # c f h w -> f c h w
            image = einops.rearrange(image, 'c f h w -> (c f) h w')
        else: 
            raise ValueError('The shape of the input is not supported.')
        image = image.transpose((1, 2, 0))  
        if self.transform:
            image = self.transform(image)
        
        image = einops.rearrange(image, '(c f) h w -> c f h w', c=2)
        target = image[0:1, ...]
        source = image[1:2, ...]
        return target, source

    def __len__(self):
        return self.file.shape[0]