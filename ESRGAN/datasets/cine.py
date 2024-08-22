import numpy as np
import glob
from torch.utils.data import Dataset
import einops

# class Cine(Dataset):
#     def __init__(self, path, transform=None):
#         super(Cine, self).__init__()
#         self.file_path = path  # 传入图像所在的文件夹路径
#         self.file = np.load(path)
#         self.transform = transform

#     def __getitem__(self, index):
#         image = self.file[index] 
#         if self.transform:
#             image = self.transform(image)  # 由于传入的是 chw totensor操作会将我的数据的转换为wch 我们需要将其转换为chw
#             image = image.permute(1,2,0)
#             target = image[0]
#             source = image[1]
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