import os
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import CIFAR10
from datasets.cine import Cine
from torch.utils.data import Subset
import numpy as np


def get_dataset(config, train=True):
    
    if config.data.dataset == "CINE":
        if train:
            my_path = config.data.training_dataset_path
            if config.data.random_flip:
                
                training_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation((-10, -10)),
                    ])
            else:
                training_transform = transforms.ToTensor()
                validation_transform = transforms.ToTensor()
            
            training_dataset = Cine(
                path=my_path,
                transform=training_transform
            )
            validation_dataset = Cine(
                path=my_path,
                transform=validation_transform
            )
            
            num_items = len(training_dataset)

            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2023)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, validation_indices = (
                indices[: int(num_items * 0.99)],
                indices[int(num_items * 0.99) :],
            )
            
            training_dataset = Subset(validation_dataset, train_indices)
            validation_dataset = Subset(training_dataset, validation_indices)
            
            
            return training_dataset, validation_dataset
        
        else:
            my_path = config.data.testing_dataset_path
            transform = transforms.ToTensor()
            testing_dataset = Cine(
                path = my_path,
                transform=transform
            )
            return testing_dataset
            


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:  # 均匀量化
        X = X / 256.0 * 255.0 + torch.rand_like(X) / 256.0
        
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:  # 数据归一化到[-1,1]
        X = 2 * X - 1.0
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, "image_mean"):
        return X - config.image_mean.to(X.device)[None, ...]
    
    return X


def inverse_data_transform(config, X):
    if hasattr(config, "image_mean"):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:  # 数据归一化到[0,1]
        X = (X + 1.0) / 2.0

    return torch.clamp(X, 0.0, 1.0)
