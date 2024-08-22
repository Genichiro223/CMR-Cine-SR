import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from datasets.cine import Cine
from torch.utils.data import Subset
import numpy as np


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)  # 57，25，128，128 （57，25）为开始裁切的左上角坐标

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(self.x1, self.x2, self.y1, self.y2)


def get_dataset(args, config, train=True):

    if config.data.dataset == "CINE":
        if train:
            my_path = config.data.training_dataset_path
            if config.data.random_flip:
                training_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation((-10, 10)),
                    ])
            else: 
                training_transform = transforms.ToTensor()
                
            validation_transform = transforms.ToTensor()
            
            training_dataset = Cine(path=my_path, transform=training_transform)
            validation_dataset = Cine(path=my_path, transform=validation_transform)
            

            num_items = len(training_dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2024)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, validation_indices = (
                indices[: int(num_items * 0.99)],
                indices[int(num_items * 0.99) :],
            )
            validation_dataset = Subset(validation_dataset, validation_indices)
            training_dataset = Subset(training_dataset, train_indices)
            
            return training_dataset, validation_dataset
        
        else:
            my_path = config.data.testing_dataset_path
            
            testing_dataset = Cine(
                    path=my_path,
                    transform=transforms.ToTensor()
                )
            return testing_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:  # false
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
    if config.data.gaussian_dequantization:  # false
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:  # true  0~1 -> -1~1
        X = 2 * X - 1.0
    elif config.data.logit_transform:  # false
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]
    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.0) / 2.0

    #return torch.clamp(X, 0.0, 1.0)
    return X
