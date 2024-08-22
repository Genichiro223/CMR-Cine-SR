import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, LSUN
from datasets.celeba import CelebA
from datasets.cine import Cine
from datasets.ffhq import FFHQ
from torch.utils.data import Subset
import numpy as np

def get_dataset(args, config, train=True):
    if config.data.random_flip is False:
        tran_transform = test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])
    else:
        tran_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config.data.image_size),
            transforms.ToTensor()
        ])

    if config.data.dataset == "CINE":
        if train:
            my_path = config.data.training_dataset_path
            if config.data.random_flip:
                dataset = Cine(
                    path=my_path,
                    transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.ToTensor()
                    ])
                )
            else:
                dataset = Cine(
                    path=my_path,
                    transform=transforms.ToTensor(),
                )

            num_items = len(dataset)
            indices = list(range(num_items))
            random_state = np.random.get_state()
            np.random.seed(2023)
            np.random.shuffle(indices)
            np.random.set_state(random_state)
            train_indices, test_indices = (
                indices[: int(num_items * 0.9)],
                indices[int(num_items * 0.9) :],
            )
            test_dataset = Subset(dataset, test_indices)
            dataset = Subset(dataset, train_indices)
            return dataset, test_dataset
        else:
            my_path = config.data.testing_dataset_path
            if config.data.random_flip:
                dataset = Cine(
                    path=my_path,
                    transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5), 
                    transforms.ToTensor()
                    ])
                )
            else:
                dataset = Cine(
                    path=my_path,
                    transform=transforms.ToTensor(),
                )
            return dataset

def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:  
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X

def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)
