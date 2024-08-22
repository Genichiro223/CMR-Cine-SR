import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

from models.diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer, utils
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from PIL import Image
import torchvision.utils as tvu
import einops
import torchvision

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd": 
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args 
        self.config = config
        if device is None:
            device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            
        self.device = device
        self.model_var_type = config.model.var_type
        
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def train(self):
        
        args, config = self.args, self.config
        tb_logger = config.tb_logger

        training_dataset, validation_dataset = get_dataset(args, config)

        training_loader = data.DataLoader(
            training_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(config, model.parameters())

        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        epoch, step = 0, 0
        
        if args.resume_training:
            states = torch.load(os.path.join(args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = config.optim.eps
            optimizer.load_state_dict(states[1])
            epoch = states[2]
            step = states[3]
            if config.model.ema:
                ema_helper.load_state_dict(states[4])
        
        while step < config.training.n_iters:
            
            data_start = time.time()
            data_time = 0
            epoch += 1

            for i, (target, source) in enumerate(training_loader):  
                # target: high resolution, source: low resolution, the noise is added
                # to the high resolution image and the source image is seen as the condition
                
                data_time += time.time() - data_start
                
                model.train()
                step += 1
                
                target = einops.rearrange(target, 'b c f h w ->  (b f) c h w' ).to(self.device)
                source = einops.rearrange(source, 'b c f h w ->  (b f) c h w' ).to(self.device) 
                    
                target = data_transform(config, target)
                source = data_transform(config, source)
                n = target.size(0)
                e = torch.randn(target.shape).to(self.device) 
                b = self.betas
                
                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, target, source, t, e, b)  # compute the loss

                tb_logger.add_scalar("loss", loss, global_step=step)

                logging.info(
                    f"epoch: {epoch}, step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )

                optimizer.zero_grad()
                loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        config.optim.grad_clip
                    )
                except Exception:
                    pass

                optimizer.step()

                if config.model.ema:
                    ema_helper.update(model)

                if step % config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(args.log_path, "ckpt.pth"))
                    
                    
                # validation
                if step % config.training.validation_freq == 0 or step == 1:
                    self.validate(model, validation_dataset, step, epoch)   
                
                if step >= self.config.training.n_iters:
                    logging.info(
                        f"Arrive at the max training steps: {config.training.n_iters}. Training is stopped."
                    )
                    break
                
                data_start = time.time()

    def validate(self, model, validation_dataset, step, epoch):
        
        args, config = self.args, self.config
        tb_logger = config.tb_logger

        random_index = np.random.randint(0, len(validation_dataset))
        
        target, source = validation_dataset[random_index]
        
        target = einops.rearrange(target, 'c f h w ->  f c h w' ).to(self.device)
        source = einops.rearrange(source, 'c f h w ->  f c h w' ).to(self.device)  # the "frame" is set as the channel dimension 
        
        target = data_transform(config, target)
        source = data_transform(config, source)
        e = torch.randn(target.shape).to(self.device)
        b = self.betas
        n = target.size(0)
        t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
        args.timestepes = 100
        with torch.no_grad():
            validation_loss = loss_registry[config.model.type](model, source, target, t, e, b)  
            m, x = self.sample_image(e, source, model, last=False)
            x = [inverse_data_transform(config, y) for y in x]
            result = x[-1]
        img_grid = torchvision.utils.make_grid(result)
        tb_logger.add_image('Denoised Image', img_grid, global_step=step)
        
        tb_logger.add_scalar("validation_loss", validation_loss.item(), global_step=step)

        logging.info(
            f"epoch: {epoch}, step: {step}, validation_loss: {validation_loss}"
        )
    
    def sample(self):
        config, args = self.config, self.args
        model = Model(config)

        if not args.use_pretrained:

            if getattr(config.sampling, "ckpt_id", None) is None:
                
                states = torch.load(
                    os.path.join(args.log_path, "ckpt.pth"),
                    map_location=config.device,
                )
            else:
                logging.info(f"loading ckpt_{config.sampling.ckpt_id}.pth")
                states = torch.load(os.path.join(args.log_path, f"ckpt_{config.sampling.ckpt_id}.pth"),
                    map_location=self.config.device,)

            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if config.model.ema:
                ema_helper = EMAHelper(mu=config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None

        model.eval()

        if args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")


    def sample_sequence(self, model):

        config, args = self.config, self.args
        
        testing_dataset = get_dataset(args, config, train=False)
                
        testing_loader = data.DataLoader(testing_dataset,
                                         batch_size=config.sampling.batch_size,
                                         num_workers=config.data.num_workers,
                                         shuffle=False)
        
        store_list = []
        
        for idx, (target, source) in enumerate(testing_loader):  # the shape of the input is batch, 1, frame, width, height
            
            # the shape of the target is batch channel frame height width
            b, c, f, h, w = source.shape
            source = einops.rearrange(source, 'b c f h w ->  (b f) c h w' ).to(self.device)  
    
            source_image = data_transform(self.config, source)
            
            # sampling from pure Gaussian noise
            initial_noise = torch.randn_like(source_image).to(self.device)
            
            with torch.no_grad():
                
                m, x = self.sample_image(initial_noise, source_image, model, last=False)  # m is the intermediate sampled xt, x is the predicted x0 
                source_image = np.stack([inverse_data_transform(config, y).to("cpu").numpy() for y in source_image])
                # source_image = source_image.to("cpu").numpy()
                x = [inverse_data_transform(config, y) for y in x]
                result = x[-1].numpy()  # last one step
                result = einops.rearrange(result, '(b f) c h w -> b c f h w', b=b, f=f)
                source_image = einops.rearrange(source_image, '(b f) c h w -> b c f h w', b=b, f=f)
                
                concat_sample = np.concatenate(
                    [source_image, result, target.numpy()], axis=1
                )
                store_list.append(concat_sample)
                np.save(os.path.join(self.args.image_folder, f"sample_batch{idx}.npy"), concat_sample)
        
        np.save(os.path.join(self.args.image_folder, f"total_batch.npy"), np.concatenate(store_list, axis=0))       
        
        
    def sample_image(self, initial_noise, source_image, model, last=True):
        
        try:  # 可能会引发异常的代码
            skip = self.args.skip  
        except Exception:  #捕获异常并进行处理
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":       
                skip = self.num_timesteps // self.args.timesteps  # 总扩散步数除以实际推理步数得到跳跃步数
                seq = range(0, self.num_timesteps, skip)  # 得到推理过程的t的值
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError

            from functions.denoising import generalized_steps
            x = generalized_steps(initial_noise, seq, model, source_image, self.betas, eta=self.args.eta)

        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps

            x = ddpm_steps(initial_noise, seq, model, source_image, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

def norma(x):
    return (x - x.min()) / (x.max() - x.min())