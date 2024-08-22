import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image
import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

class Cine(Dataset):
    def __init__(self, path, transform=None):
        super(Cine, self).__init__()
        self.file_path = path  # 传入图像所在的文件夹路径
        self.file = np.load(path)
        self.transform = transform

    def __getitem__(self, index):
        image = self.file[index] 
        
        if self.transform:
            image = self.transform(image)
            image = image.permute(1,2,0)
            image = image.float()
            source, target = torch.chunk(image, chunks=2, dim=0)
        return target, source

    def __len__(self):
        return self.file.shape[0]