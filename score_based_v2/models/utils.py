import torch
import sde_lib
import numpy as np
import torch
import os
import logging

_MODELS = {}
def register_model(cls=None, *, name=None):
  """A decorator for registering model classes.""" 
  def _register(cls):  
    local_name = cls.__name__ if name is None else name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls
  return _register if cls is None else _register(cls)

def get_model(name):
  return _MODELS[name]

def create_model(config):
  """Create the score model."""
  model_name = config.model.name  # 'ncsn'
  score_model = get_model(model_name)(config)
  score_model = score_model.to(config.device)
  score_model = torch.nn.DataParallel(score_model)
  return score_model

def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), 
                np.log(config.model.sigma_min), 
                config.model.num_scales))  # 从最大噪声到最小噪声，等比例取1000num_scalses个噪声

  return sigmas

def get_model_fn(model, train=False):  # 得到score-based model的输出
  """Create a function to give the output of the score-based model.

  Args:
    model: The score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels):  # 传入输入和标签
    """Compute the output of the score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models. 

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      print(x.shape, labels.shape)
      return model(x, labels)

  return model_fn


def get_score_fn(sde, model, train=False, continuous=False):
  
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """
  model_fn = get_model_fn(model, train=train)  # 返回 model 的函数接口
  
  if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
    def score_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, sde_lib.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels)  # 基于x和时间t来计算score
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, sde_lib.VESDE):  # 如果是VESDE

    def score_fn(x, t):  # 传入数据和采样得到的时间t，这里的t是用来索引方差的
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]  # sde对象中的marginal_prob方法返回对应时间下的均值和方差，这里取方差
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)  # 将数据和方差传入模型，这里的x是拼接好的target x和 source y，返回模型的计算结果
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def restore_checkpoint(ckpt_dir, state, device):
  if not os.path.exists(ckpt_dir):
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:  
    loaded_state = torch.load(ckpt_dir, map_location=device)  
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['epoch'] = loaded_state['epoch']
    state['step'] = loaded_state['step']
    return state
  
def save_checkpoint(ckpt_dir, state):  
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'epoch': state['epoch'],
    'step' : state['step']
  }
  torch.save(saved_state, ckpt_dir)