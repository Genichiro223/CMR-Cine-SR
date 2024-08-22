import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from datasets.Cine import Cine
import numpy as np

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):  # 将数据裁切并放缩到给定的分辨率
  """Crop and resize an image to the given resolution."""
  crop = min(image.shape[0], image.shape[1])  # 这里假定输入的图像不是正方形，为此选定最小的边作为裁切的目标边长
  h, w = image.shape[0], image.shape[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]  # 保证切出来的图像是正方形
  image = transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC ,antialias=True)(image)
  return image


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = torch.round(h * ratio).int()
  w = torch.round(w * ratio).int()
  return transforms.Resize([h, w], antialias=True)(image)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return transforms.functional.crop(image, top, left, size, size)


def get_dataset(config, uniform_dequantization=False, evaluation=False):

  batch_size = config.evalu.batch_size if evaluation else config.training.batch_size
  
  if batch_size % torch.cuda.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by the number of devices ({torch.cuda.device_count()})')


  num_epochs = None if not evaluation else 1

  if config.data.dataset == "cine":
    my_path = config.training.dataset_path
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

  else:
      dataset, test_dataset = None, None
      
  return dataset, test_dataset