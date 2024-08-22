import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from sde_lib import VESDE, VPSDE, subVPSDE


def get_optimizer(config, params):
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, 
                           lr=config.optim.lr, 
                           betas=(config.optim.beta1, 0.999), 
                           eps=config.optim.eps, weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')
  return optimizer


def optimization_manager(config):
  
  """Returns an optimize_fn based on `config`."""
  
  def optimize_fn(optimizer, 
                  params, 
                  step, 
                  lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip
                  ):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step/warmup, 1.0)
        
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
      
    optimizer.step()
    
  return optimize_fn


def get_sde_loss_fn(sde, 
                    train, 
                    reduce_mean=True, 
                    continuous=True, 
                    likelihood_weighting=True, 
                    eps=1e-5):

  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)  # the default is False, sum the loss across data dimensions

  def loss_fn(model, x, y):
    """Compute the loss function.

    Args:
      model: A score model.
      x: The target(high-resolution image).
      y: The source(low-resolution image).
    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    
    score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
    
    t = torch.rand(x.shape[0], device=x.device) * (sde.T - eps) + eps
    z = torch.randn_like(x)  # sampling random variable z from the normal distribution 
    mean, std = sde.marginal_prob(x, t)  # return the x_{0} and \sigma_{t}
    x_t = mean + std[:, None, None, None] * z
    concat_input = torch.concatenate([x_t, y], dim=1)
    
    score = score_fn(concat_input, t)  # the shape of the input should be [N, 2, H, W] and the output shape is [N, 1, H, W]

    if not likelihood_weighting:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      g2 = sde.sde(torch.zeros_like(x), t)[1] ** 2
      losses = torch.square(score + z / std[:, None, None, None])
      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2

    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_smld_loss_fn(vesde, train, reduce_mean=False):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))  # 沿着给定维度对tensor反转
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, x, y):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (x.shape[0],), device=x.device)  # 在0～N之间随机采样产生对应时间t的label, batch.shape[0]-> batch_size
    
    sigmas = smld_sigma_array.to(x.device)[labels]
    noise = torch.randn_like(x) * sigmas[:, None, None, None]
    x_t = noise + x
    
    concat_input = torch.concatenate([x_t, y], dim=1)
    score = model_fn(concat_input, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(vpsde, train, reduce_mean=True):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, x, y):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (x.shape[0],), device=y.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(x.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(x.device)
    noise = torch.randn_like(x)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * x + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    concat_input = torch.concatenate([perturbed_data, y], dim=1)
    score = model_fn(concat_input, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn



def get_step_fn(sde, 
                train, 
                optimize_fn=None, 
                reduce_mean=False, 
                continuous=True, 
                likelihood_weighting=False):
  
  """
  Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. 
                  Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  
  if continuous:  
    loss_fn = get_sde_loss_fn(sde, 
                              train, 
                              reduce_mean=reduce_mean,
                              continuous=True, 
                              likelihood_weighting=likelihood_weighting)
  else:
    assert not likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(sde, train, reduce_mean=reduce_mean)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(sde, train, reduce_mean=reduce_mean)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn(state, x, y):
    """Running one step of training or evaluation.

    Args:
      state: A dictionary of training information, containing the score model, optimizer, EMA status, and number of optimization steps.
      x: The target image.
      y: The source image.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']  
      optimizer.zero_grad()  
      loss = loss_fn(model, x, y)
      loss.backward() 
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, x, y)
        ema.restore(model.parameters())

    return loss

  return step_fn
