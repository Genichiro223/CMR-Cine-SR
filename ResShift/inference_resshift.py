
import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import ResShiftSampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument('--cfg_path', type=str, default='', help='Path of the config file.')
    parser.add_argument("--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("--steps", type=int, default=15, help="Diffusion length.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    args = parser.parse_args()

    return args

def get_configs(args):
    configs = OmegaConf.load(args.cfg_path)

    configs.diffusion.params.steps = args.steps

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)


    return configs

def main():
    args = get_parser()

    configs = get_configs(args)

    resshift_sampler = ResShiftSampler(
            configs,
            seed=args.seed,
            )
    resshift_sampler.inference(args.out_path, bs=16, noise_repeat=False)

if __name__ == '__main__':
    main()
