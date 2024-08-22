import os, sys, math, random

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from omegaconf import OmegaConf

from utils import util_net
from utils import util_image
from utils import util_common

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

from datapipe.datasets import create_dataset


class BaseSampler:
    def __init__(
            self,
            configs,
            sf=None,
            seed=10000,
            ):
        '''
        Input:
            configs: config, see the yaml file in folder ./configs/
            sf: int, super-resolution scale
            seed: int, random seed
        '''
        self.configs = configs
        self.seed = seed

        if sf is None:
            sf = configs.diffusion.params.sf
        self.sf = sf

        self.setup_dist()  # setup distributed training: self.num_gpus, self.rank

        self.setup_seed()

        self.build_model()

    def setup_seed(self, seed=None):
        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_dist(self, gpu_id=None):
        num_gpus = torch.cuda.device_count()
        assert num_gpus == 1, 'Please assign one available GPU using CUDA_VISIBLE_DEVICES!'
        self.num_gpus = num_gpus
        self.rank = int(os.environ['LOCAL_RANK']) if num_gpus > 1 else 0

    def write_log(self, log_str):
        if self.rank == 0:
            print(log_str)

    def build_model(self):
        # diffusion model
        log_str = f'Building the diffusion model with length: {self.configs.diffusion.params.steps}...'
        self.write_log(log_str)
        self.base_diffusion = util_common.instantiate_from_config(self.configs.diffusion)
        
        model = util_common.instantiate_from_config(self.configs.model).cuda()
        ckpt_path = self.configs.model.test_ckpt_path
        assert ckpt_path is not None
        
        self.write_log(f'Loading Diffusion model from {ckpt_path}...')
        self.load_model(model, ckpt_path)
        
        self.model = model.eval()

        # autoencoder model
        if self.configs.autoencoder is not None:
            ckpt_path = self.configs.autoencoder.ckpt_path
            assert ckpt_path is not None
            self.write_log(f'Loading AutoEncoder model from {ckpt_path}...')
            autoencoder = util_common.instantiate_from_config(self.configs.autoencoder).cuda()
            self.load_model(autoencoder, ckpt_path)
            autoencoder.eval()
            if self.configs.autoencoder.use_fp16:
                self.autoencoder = autoencoder.half()
            else:
                self.autoencoder = autoencoder
        else:
            self.autoencoder = None

    def load_model(self, model, ckpt_path=None):
        state = torch.load(ckpt_path, map_location=f"cuda:{self.rank}")
        if 'state_dict' in state:
            state = state['state_dict']
        util_net.reload_model(model, state)

class ResShiftSampler(BaseSampler):
    def sample_func(self, y0, noise_repeat=False):
        '''
        Input:
            y0: n x c x h x w torch tensor, low-quality image, [-1, 1], RGB
        Output:
            sample: n x c x h x w, torch tensor, [-1, 1], RGB
        '''   

        if noise_repeat:  
            self.setup_seed()

        model_kwargs = {'lq':y0,} if self.configs.model.params.cond_lq else None 
        
        # model_kwargs = {}
        
        results = self.base_diffusion.p_sample_loop(
                y=y0,
                model=self.model,
                first_stage_model=self.autoencoder,
                noise=None,
                noise_repeat=noise_repeat,
                clip_denoised=False,
                denoised_fn=None,
                model_kwargs=model_kwargs,
                progress=False,
                )    # This has included the decoding for latent space

        return results
    
    @torch.no_grad()
    def inference(self, out_path, bs=1, noise_repeat=False):
        '''
        Inference demo.
        Input:
            in_path: str, folder or image path for LQ image
            out_path: str, folder save the results
            bs: int, default bs=1, bs % num_gpus == 0
        '''
        def _process_per_image(im_lq_tensor):
            '''
            Input:
                im_lq_tensor: b x c x h x w, torch tensor, [0,1], RGB
            Output:
                im_sr: h x w x c, numpy array, [0,1], RGB
            '''

            # if im_lq_tensor.shape[2] > self.chop_size or im_lq_tensor.shape[3] > self.chop_size:
            #     im_spliter = ImageSpliterTh(
            #             im_lq_tensor,
            #             self.chop_size,
            #             stride=self.chop_stride,
            #             sf=self.sf,
            #             extra_bs=self.chop_bs,
            #             )
            #     for im_lq_pch, index_infos in im_spliter:
            #         # print(im_lq_pch.shape)
            #         im_sr_pch = self.sample_func(
            #                 (im_lq_pch - 0.5) / 0.5,
            #                 noise_repeat=noise_repeat,
            #                 )     # 1 x c x h x w, [-1, 1]
            #         im_spliter.update(im_sr_pch, index_infos)
            #     im_sr_tensor = im_spliter.gather()
            # else:
            im_sr_tensor = self.sample_func(
                    (im_lq_tensor - 0.5) / 0.5,
                    noise_repeat=noise_repeat,
                    )     # 1 x c x h x w, [-1, 1] 传入sample_func的数据是-1到1 
            
            im_sr_tensor = im_sr_tensor * 0.5 + 0.5
            return im_sr_tensor

        #in_path = Path(in_path) if not isinstance(in_path, Path) else in_path  # in_path 字符串路径
        out_path = Path(out_path) if not isinstance(out_path, Path) else out_path
        if not out_path.exists():
            out_path.mkdir(parents=True)

            
        data_config = self.configs.data.get('test')
        dataset = create_dataset(data_config)
        
        self.write_log(f'The size of testing dataset is: {len(dataset)}.')
        
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=40,
                shuffle=False,
                drop_last=False,
                )
        idx = 0
        
        with torch.no_grad():
            store_list = []
            for micro_data in dataloader:
                results = _process_per_image(micro_data['lq'].cuda())    # b x h x w x c, [0, 1], RGB
                concat_result = np.concatenate([micro_data['lq'].detach().cpu().numpy(), results.detach().cpu().numpy(), micro_data['gt'].detach().cpu().numpy()], axis=1)
                np.save(str(out_path) + f"/concat_result{idx}.npy", concat_result)
                idx+=1
                for jj in range(results.shape[0]):
                    im_sr = util_image.tensor2img(results[jj], rgb2bgr=True, min_max=(0.0, 1.0))
                    # im_name = Path(micro_data['path'][jj]).stem
                    im_name = f"test{idx}_{jj}"
                    im_path = out_path / f"{im_name}.png"
                    util_image.imwrite(im_sr, im_path, chn='bgr', dtype_in='uint8')
                    
            np.save(os.path.join(out_path, f"resshift_finaltest.npy"), np.concatenate(store_list, axis=0))

def rescaled(x):
    return x * 2. - 1.   # transofrm [0, 1] to [-1, 1]