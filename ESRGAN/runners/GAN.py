import torch
import torchvision
from functions.utils import gradient_penalty, AverageMeter
from functions import get_optimizer
from functions.vgg_loss import VGGLoss
import os
import glob
import numpy as np
import logging
from models.GAN_model import *
import torch.utils.data as data
from datasets import get_dataset, data_transform, inverse_data_transform
from models.GAN_model import Generator, Discriminator
import time
from PIL import Image
import tqdm
import natsort as ns
import einops

def norm(x):
    return (x - x.min()) / (x.max() - x.min())


def _extracted_from_train(optimizer, loss, loss_store):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_store.update(loss.item())


class GAN(object):
    
    def __init__(self, args, config, device=None): 
        self.args = args
        self.config = config
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    def _load_states(self, model, optimizer, ckpt_path):  
        states = torch.load(ckpt_path)   
        model.load_state_dict(states[0])
        states[1]["param_groups"][0]["eps"] = self.config.optim.eps
        optimizer.load_state_dict(states[1])
        start_epoch = states[2]   
        step = states[3]
        return start_epoch, step
    
    def train(self):     

        args, config = self.args, self.config  
        tb_logger = config.tb_logger
        training_dataset, validation_dataset = get_dataset(config)       
         
        training_loader = data.DataLoader(training_dataset,
                                          batch_size=config.training.batch_size,
                                          shuffle=True,
                                          num_workers=config.data.num_workers)  
        
        gen, disc = Generator().to(self.device), Discriminator().to(self.device)  
        gen, disc = torch.nn.DataParallel(gen), torch.nn.DataParallel(disc)
        
        opt_gen = get_optimizer(config, gen.parameters())
        opt_disc = get_optimizer(config, disc.parameters())
        
        GEN_LOSS_train = AverageMeter()
        DISC_LOSS_train = AverageMeter()  
        PSNR_train = AverageMeter()
        SSIM_train = AverageMeter()

        epoch, step = 0, 0

        if args.resume_training:
            gen_states = torch.load(os.path.join(args.log_path, "gen_ckpt.pth"))
            gen.load_state_dict(gen_states[0])
            gen_states[1]["param_groups"][0]["eps"] = config.optim.eps
            opt_gen.load_state_dict(gen_states[1])
            
            disc_states = torch.load(os.path.join(args.log_path, "disc_ckpt.pth"))   
            disc.load_state_dict(disc_states[0])
            disc_states[1]["param_groups"][0]["eps"] = config.optim.eps
            opt_disc.load_state_dict(disc_states[1])    
            
            epoch = gen_states[2]   
            step = gen_states[3]

        l1 = torch.nn.L1Loss().to(self.device)  
        vgg_loss = VGGLoss().to(self.device) 
        
        while step < config.training.n_iters:
            
            data_start = time.time()
            data_time = 0     
            epoch += 1  
            
            for idx, (target, source) in enumerate(training_loader):  # the dimension of the target/source is (B, 1, F, H, W)
                step += 1
                data_time += time.time() - data_start
                gen.train()
                disc.train()

                target = einops.rearrange(target, 'b c f h w ->  (b f) c h w' ).to(self.device).float()
                source = einops.rearrange(source, 'b c f h w ->  (b f) c h w' ).to(self.device).float()
                
                l1_loss, adversarial_loss, loss_for_vgg = 0, 0, 0  # set the initial loss value to be zero
                
                gen_img = gen(source)  
                disc_fake = disc(gen_img)  
                    
                label = target

                l1_loss +=  1e-2 * l1(gen_img, label)
                adversarial_loss += 5e-3 * -torch.mean(disc_fake)
                loss_for_vgg += vgg_loss(gen_img, label)
                gen_loss = l1_loss + adversarial_loss + loss_for_vgg
                tb_logger.add_scalar("generator_loss", gen_loss, global_step=step)

                _extracted_from_train(opt_gen, gen_loss, GEN_LOSS_train)
                
                # save the states of the generator
                if step % config.training.snapshot_freq == 0 or step == 1:
                    gen_states = [gen.state_dict(), opt_gen.state_dict(), epoch, step,]
                    torch.save(gen_states, os.path.join(args.log_path, f"gen_ckpt_{step}.pth"),)
                    torch.save(gen_states, os.path.join(args.log_path, "gen_ckpt.pth"))
                             
                # train the discriminator
                loss_critic = 0
                    
                label = target
                LR_img = source
                    
                fake = gen(LR_img)
                critic_real = disc(label)  
                critic_fake = disc(fake.detach())
                    
                gp = gradient_penalty(disc, label, fake, device=self.device)
                    
                loss_critic += (-(torch.mean(critic_real) - torch.mean(critic_fake)) + config.loss.lambda_gp * gp)
                tb_logger.add_scalar("discriminator_loss", loss_critic, global_step=step)
                
                logging.info(f"epoch: {epoch}, step: {step}, generator_loss: {gen_loss.item()}, discriminator_loss: {loss_critic.item()}")
                
                _extracted_from_train(opt_disc, loss_critic, DISC_LOSS_train)
        
        
                if step % config.training.snapshot_freq == 0 or step == 1:
                    gen_states = [gen.state_dict(), opt_gen.state_dict(), epoch, step,]
                    torch.save(gen_states, os.path.join(args.log_path, f"gen_ckpt_{step}.pth"),)
                    torch.save(gen_states, os.path.join(args.log_path, "gen_ckpt.pth"))
                    
                    disc_states = [disc.state_dict(), opt_disc.state_dict(), epoch, step,]
                    torch.save(disc_states, os.path.join(args.log_path, f"disc_ckpt_{step}.pth"),)
                    torch.save(disc_states, os.path.join(args.log_path, "disc_ckpt.pth"))
                
                if step % config.training.validation_freq == 0 or step==1:
                    self.validate(gen, disc, validation_dataset, step, epoch)
                    
                if step > config.training.n_iters:
                    logging.info("The step is arrived at the max training step, training completed.")
                    break
                
        return GEN_LOSS_train.avg, DISC_LOSS_train.avg 
    
    def validate(self, gen, disc, validation_dataset, step, epoch):
        
        args, config = self.args, self.config
        tb_logger = config.tb_logger

        validation_dataloader = data.DataLoader(validation_dataset,
                                             batch_size=config.training.batch_size,
                                             shuffle=True,
                                             num_workers=config.data.num_workers,)
        
        l1 = torch.nn.L1Loss().to(self.device)  
        vgg_loss = VGGLoss().to(self.device)
        gen_val_loss, disc_val_loss = 0.0, 0.0
        
        batch_count = 0
        
        for idx, (target, source) in enumerate(validation_dataloader):
                
                gen.eval()
                disc.eval()
                target = einops.rearrange(target, 'b c f h w ->  (b f) c h w' ).to(self.device).float()
                source = einops.rearrange(source, 'b c f h w ->  (b f) c h w' ).to(self.device).float()

                l1_loss, adversarial_loss, loss_for_vgg = 0, 0, 0
                
                with torch.no_grad():
                    gen_img = gen(source)  
                    disc_fake = disc(gen_img)  
                    
                    label = target

                    l1_loss +=  1e-2 * l1(gen_img, label)
                    adversarial_loss += 5e-3 * -torch.mean(disc_fake)
                    loss_for_vgg += vgg_loss(gen_img, label)
                    gen_loss = l1_loss + adversarial_loss + loss_for_vgg
                    
                    gen_val_loss += gen_loss

                    loss_critic = 0
                        
                    label = target
                    LR_img = source
                        
                    fake = gen(LR_img)
                    critic_real = disc(label)  
                    critic_fake = disc(fake.detach())
                    
                    torch.set_grad_enabled(True)
                    gp = gradient_penalty(disc, label, fake, device=self.device)
                    torch.set_grad_enabled(False)
                                
                    loss_critic += (-(torch.mean(critic_real) - torch.mean(critic_fake)) + config.loss.lambda_gp * gp)

                    batch_count += 1
                
        average_gen_loss = gen_val_loss / batch_count
        average_disc_loss = disc_val_loss / batch_count
        
        random_index = np.random.randint(0, target.shape[0])
        pred = [source[random_index], fake[random_index], target[random_index]]
        img_grid = torchvision.utils.make_grid(pred)
        
        tb_logger.add_image('LR, SR and HR', img_grid, global_step=epoch)
        tb_logger.add_scalar("generator_val_loss", average_gen_loss, global_step=step)
        tb_logger.add_scalar("discriminator_val_loss", average_disc_loss, global_step=step)
            
        logging.info(f"epoch: {epoch}, step: {step}, generator_val_loss: {average_gen_loss}, discriminator_val_loss: {average_disc_loss}")
    

    def sample(self):
        config, args = self.config, self.args
        gen = Generator()
        gen_states = torch.load(
            os.path.join(self.args.log_path, "gen_ckpt.pth"),
            map_location=self.config.device,
        )
        gen = gen.to(self.device)
        gen = torch.nn.DataParallel(gen)
        gen.load_state_dict(gen_states[0], strict=True)
        gen.eval()

        testing_dataset = get_dataset(config, train=False)
        testing_loader = data.DataLoader(testing_dataset, 
                                         batch_size = config.sampling.batch_size,
                                         num_workers = config.data.num_workers)
        
        store_list = []
        
        for idx, (source, target) in tqdm.tqdm(enumerate(testing_loader)):  
            
            source, target = torch.unsqueeze(source.to(self.device), dim=1), torch.unsqueeze(target.to(self.device), dim=1)
            sr_sample = gen(source.float()).detach().to('cpu').numpy()  # 将样本传入模型 并将输出结果转移到cpu上
            concat_sample = np.concatenate((source.to('cpu').numpy(), sr_sample, target.to('cpu').numpy()), axis=1)  # 将三个样本拼接起来
            store_list.append(concat_sample)
            if idx % config.sampling.shot == 0:  # 没执行shot步，对结果保存一次
                store_array = np.concatenate(store_list, axis=0)
                np.save(os.path.join(self.args.image_folder, f"{idx}.npy"), store_array)
        
        np.save(os.path.join(self.args.image_folder, f"GAN_final.npy"), store_array)