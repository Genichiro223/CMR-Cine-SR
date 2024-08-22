import gc
import io
import os
import time
import numpy as np
import logging
import shutil
import torch.utils.data as data
# Keep the import below for registering all model definitions
from models import  ncsnv2 #, ncsnpp
# from models.attention_unetbackbone import Model
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import evaluation
import likelihood
import sde_lib
from absl import flags 
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from models.utils import save_checkpoint, restore_checkpoint
FLAGS = flags.FLAGS


def train(config, workdir): 
  """Runs the training pipeline.
  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  os.makedirs(sample_dir, exist_ok=True)
  
  tb_dir = os.path.join(workdir, "tensorboard")
  os.makedirs(tb_dir, exist_ok=True)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  model = mutils.create_model(config)
  #model = model(config)
  model = model.to(config.device) 
  
  ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)  # 0.999 cheacked
  optimizer = losses.get_optimizer(config, model.parameters())  
  state = dict(optimizer=optimizer, model=model, ema=ema, epoch=0, step=0)  # create a dictionary to state the optimizer, model, ema, epoch, step
  
  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  os.makedirs(checkpoint_dir, exist_ok=True)  
   
  # Resume training when intermediate checkpoints are detected 
  # or start from scratch if no checkpoints are found.
  state = restore_checkpoint(os.path.join(workdir, "checkpoints", "checkpoint.pth"), state, config.device)  
  initial_epoch = int(state['epoch'])
  step = initial_step = int(state['step'])
  
  # Build data iterators
  train_ds, eval_ds = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
  train_loader = data.DataLoader(dataset=train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=0)
  eval_loader = data.DataLoader(dataset=eval_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=0)
  
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min,
                        beta_max=config.model.beta_max, 
                        N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, 
                           beta_max=config.model.beta_max, 
                           N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, 
                        sigma_max=config.model.sigma_max, 
                        N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  
  train_step_fn = losses.get_step_fn(sde,
                                     train=True, 
                                     optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, 
                                     continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)

  eval_step_fn = losses.get_step_fn(sde, 
                                    train=False,
                                    optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, 
                                    continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)


  num_epochs = config.training.epochs 
  logging.info(f"Starting training loop at epoch: {initial_epoch}, step: {initial_step}.")

  for epoch in range(initial_epoch, num_epochs + 1):
    
    state["epoch"] = epoch
    
    for i, (x, y) in enumerate(train_loader, start=1):  #  x denotes the high-resolution image and y denotes the corresponding low-resolution image
      step += 1
      x, y = x.to(config.device).float(), y.to(config.device).float()
      x, y = scaler(x), scaler(y)  # Transform data to -1~1.
      loss = train_step_fn(state, x, y)  # Execute one training step
      
      if step % config.training.log_freq == 0:
        logging.info(f"epoch: {epoch}, step: {step}, training_loss: {loss.item()}")
        writer.add_scalar("training_loss", scalar_value=loss, global_step=step)

      if step % config.training.snapshot_freq == 0 or epoch == num_epochs or step==1:
        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{step}.pth'), state)
        save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint.pth'), state)

def sample(config,
             workdir,
             sampling_folder="sampling"):

  # Create directory to store sampling results.
  sampling_dir = os.path.join(workdir, sampling_folder)
  if os.path.exists(sampling_dir):
    overwrite = False
    response = input('The sampling dictionary exists, overwrite? (Y/N)')
    if response.upper() == 'Y':
      overwrite = True
    if overwrite:
      shutil.rmtree(sampling_dir)
      os.makedirs(sampling_dir)     
  else:  
    os.makedirs(sampling_dir)  
    
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)  
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Build data pipeline
  # ---------- A simple version to test the sampling method ----------
  source = np.load(config.sampling.dataset_path)
  source = scaler(source)  # transform data to -1~1
  source = np.expand_dims(source, axis=1)
  source = torch.from_numpy(source).to(config.device)

  # Initialize model
  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, epoch=0, step=0)
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  state = restore_checkpoint(os.path.join(checkpoint_dir, "checkpoint.pth"), state, config.device)
  score_model = state['model']

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = source.shape
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  # Generate samples 
  samples, n = sampling_fn(score_model, source)
  samples = samples.cpu().numpy()

  # Write samples to disk or Google Cloud Storage
  np.save(os.path.join(sampling_dir, 'samples.npy'), samples)