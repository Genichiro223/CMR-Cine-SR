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
        self.file_path = path 
        self.file = torch.from_numpy(np.load(path)[:20])  # 20, 2, 30, 128, 128

    def __getitem__(self, index): 
        image = self.file[index]  # 2, 30, 128, 128
        # image = image.transpose((1, 2, 0))  # 128, 128, 2
        # target = image[...,0]
        # source = image[...,1]
        # if self.transform:
        #     target = self.transform(target)
        #     source = self.transform(source)
        # if self.transform:
        #     image = self.transform(image)
        target = image[0]
        source = image[1]
        return target, source

    def __len__(self):
        return self.file.shape[0]
    
    
def test_cine_dataset():
    # 创建一个临时的 numpy 文件用于测试
    test_data = np.random.rand(20, 2, 30, 128, 128).astype(np.float32)
    np.save('test_data.npy', test_data)

    # 定义一个简单的 transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 创建 Cine 数据集实例
    dataset = Cine('test_data.npy', transform=transform)

    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 迭代数据集并打印一些信息
    for i, (target, source) in enumerate(dataloader):
        print(f'Batch {i+1}')
        print(f'Target shape: {target.shape}')
        print(f'Source shape: {source.shape}')
        if i == 2:  # 只打印前三个批次
            break

    # 删除临时文件
    os.remove('test_data.npy')

if __name__ == "__main__":
    test_cine_dataset()