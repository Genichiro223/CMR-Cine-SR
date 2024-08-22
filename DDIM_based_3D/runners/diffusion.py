import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

# from models.diffusion import Model
from models.SpatialTemporal_Diffusion import Model
from models.ema import EMAHelper
from functions import get_optimizer, utils
from functions.losses import loss_registry
from datasets import get_dataset, data_transform, inverse_data_transform
from functions.ckpt_util import get_ckpt_path
from PIL import Image
import torchvision.utils as tvu

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
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
            
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
        tb_logger = self.config.tb_logger

        training_dataset, testing_dataset = get_dataset(args, config)

        training_loader = data.DataLoader(
            training_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )

        model = Model(config)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        optimizer = get_optimizer(self.config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        start_epoch, step = 0, 0
        
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
            model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(states[1])
            start_epoch = states[2]
            step = states[3]
            if self.config.model.ema:
                ema_helper.load_state_dict(states[4])

        for epoch in range(start_epoch, self.config.training.n_epochs):
            data_start = time.time()
            data_time = 0

            for i, (x, y) in enumerate(training_loader):  # x is the high resolution image, y is the low resolution image

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1
                x, y = x.to(self.device), y.to(self.device)  
                x = data_transform(self.config, x).unsqueeze(1)
                y = data_transform(self.config, y).unsqueeze(1)  # 0-1 -> -1-1
                # x = x.unsqueeze(1)
                # y = y.unsqueeze(1)
                e = torch.randn(x.shape).to(self.device) 
                b = self.betas
                
                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

                loss = loss_registry[config.model.type](model, x, y, t, e, b)  # compute the loss

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

                if self.config.model.ema:
                    ema_helper.update(model)

                if step % self.config.training.snapshot_freq == 0 or step == 1:
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.args.log_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.args.log_path, "ckpt.pth"))
                    
                    
                # validation
                if step % self.config.training.validation_freq == 0:
                    self.validate(model, testing_dataset, step, epoch)   
                
                data_start = time.time()

    def validate(self, model, testing_dataset, step, epoch):
        
        args, config = self.args, self.config
        tb_logger = self.config.tb_logger

        testing_dataloader = data.DataLoader(
            testing_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        total_validation_loss = 0.0
        batch_count = 0
        for i, (x, y) in enumerate(testing_dataloader):

                n = x.size(0)
                model.eval()
                x, y = x.to(self.device), y.to(self.device)  
                x = data_transform(self.config, x).unsqueeze(1)
                y = data_transform(self.config, y).unsqueeze(1)
                # x = x.unsqueeze(1)
                # y = y.unsqueeze(1)
                e = torch.randn(x.shape).to(self.device) 
                b = self.betas
                
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                
                with torch.no_grad():
                    validation_loss = loss_registry[config.model.type](model, x, y, t, e, b)  # compute the loss
                total_validation_loss += validation_loss.item()
                batch_count += 1
                
        average_validation_loss = total_validation_loss / batch_count
        tb_logger.add_scalar("validation_loss", average_validation_loss, global_step=step)
        logging.info(
            f"epoch: {epoch}, step: {step}, validation_loss: {average_validation_loss}"
        )
    
    def sample(self):
        model = Model(self.config)

        if not self.args.use_pretrained:

            if getattr(self.config.sampling, "ckpt_id", None) is None:
                # 读取args中输入的log_path
                states = torch.load(
                    os.path.join(self.args.log_path, "ckpt.pth"),
                    map_location=self.config.device,
                )
            else:
                logging.info(f"loading ckpt_{self.config.sampling.ckpt_id}.pth")
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location=self.config.device,
                )

            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            model.load_state_dict(states[0], strict=True)

            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(states[-1])
                ema_helper.ema(model)
            else:
                ema_helper = None


        model.eval()

        if self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_sequence(self, model):

        config = self.config
        args = self.args
        
        testing_dataset = get_dataset(args, config, train=False)
                
        testing_loader = data.DataLoader(testing_dataset,
                                       batch_size=config.sampling.batch_size,
                                       num_workers=config.data.num_workers,
                                       shuffle=False)
        
        store_list = []
        
        for idx, (target, source) in tqdm.tqdm(enumerate(testing_loader)):
            
            source, target = torch.unsqueeze(source.to(self.device), dim=1), torch.unsqueeze(target.to(self.device), dim=1)
            source_image = data_transform(self.config, source)
            # source_image = source
            initial_noise = torch.randn_like(source_image).to(self.device)
            
            with torch.no_grad():
                # 调用sample_image
                print(initial_noise.shape, source_image.shape)
                m, x = self.sample_image(initial_noise, source_image, model, last=False)  # m is the intermediate sampled xt, x is the predicted x0 
                source_image = np.stack([inverse_data_transform(config, y).to("cpu").numpy() for y in source_image])
                # source_image = source_image.to("cpu").numpy()
                x = [inverse_data_transform(config, y) for y in x]
                result = x[-1].numpy()  # last one step
                concat_sample = np.concatenate([source_image, result, target.to("cpu").numpy()], axis=1)
                # concat_sample = np.concatenate([source_image.to("cpu").numpy(), result, target.to("cpu").numpy()], axis=1)
                #concat_sample = np.concatenate([source_image.to("cpu").numpy(), result], axis=1)
                store_list.append(concat_sample)
                np.save(os.path.join(self.args.image_folder, f"sample_test{idx}.npy"), concat_sample)
        
        np.save(os.path.join(self.args.image_folder, f"DDIM_finaltest.npy"), np.concatenate(store_list, axis=0))       
        
        
    def sample_image(self, initial_noise, source_image, model, last=True):

        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
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

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

def norma(x):
    return (x - x.min()) / (x.max() - x.min())