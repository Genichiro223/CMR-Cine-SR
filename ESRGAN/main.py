# Import necessary modules
import argparse
import os
import logging
import traceback
import torch
import sys
import yaml
import shutil
import numpy as np
from models.GAN_model import Generator, Discriminator
from runners.GAN import GAN
import torch.utils.tensorboard as tb

def dict2namespace(config):  # 将字典转换为命名空间
    namespace = argparse.Namespace()  # 创建一个命名空间
    for key, value in config.items():
        new_value = dict2namespace(value) if isinstance(value, dict) else value   
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():  # 解析命令行参数和配置文件
    parser = argparse.ArgumentParser()  # 创建一个解析对象
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed"
    )
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",  # 如果在命令行中输入了--ni,
        help="No interaction. Suitable for Slurm Job launcher",
    )
    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)  # 创建新的路径用于保存日志

    # 解析config文件
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    tb_path = os.path.join(args.exp, "tensorboard", args.doc)  # tensorboard的路径
    
    if not args.sample:
        if not args.resume_training:
            if os.path.exists(args.log_path):
                overwrite = False
                if args.ni:
                    overwrite = True
                else:
                    response = input("Folder already exists. Overwrite? (Y/N)")
                    if response.upper() == "Y":
                        overwrite = True
                
                if overwrite:
                    shutil.rmtree(args.log_path)
                    shutil.rmtree(tb_path)
                    os.makedirs(args.log_path)
                    if os.path.exists(tb_path):
                        shutil.rmtree(tb_path)
                else:
                    print("Folder exists. Program halted.")
                    sys.exit(0)
            else:
                os.makedirs(args.log_path)
                
            with open(os.path.join(args.log_path, "config.yml"), "w") as f:
                yaml.dump(new_config, f, default_flow_style=False)
                
        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)  # 创建tensorboard的日志文件
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f"level {args.verbose} not supported")

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)
        
    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f"level {args.verbose} not supported.")

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
        )
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:  # 如果要执行采样操作
            os.makedirs(os.path.join(args.exp, "image_samples"), exist_ok=True)  # 在exp下创建一个文件夹，名字为image_samples
            args.image_folder = os.path.join(  # args.image_folder的路径为exp/image_samples/images
                args.exp, "image_samples", args.image_folder
            )  
            if not os.path.exists(args.image_folder):  # 如果image_folder的路径不存在，则创建
                os.makedirs(args.image_folder)
            else:  # 如果
                overwrite = False
                if args.ni:  # 如果不需要交互
                    overwrite = True  # 则覆盖
                else:
                    response = input(
                        f"Image folder {args.image_folder} already exists. Overwrite? (Y/N)"
                    )
                    if response.upper() == "Y":
                        overwrite = True
                
                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)
                else:
                    print("Output image folder exists. Program halted.")
                    sys.exit(0)


    # add device
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    logging.info(f"Using device {device}")
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():  # 如果可以使用GPU
        torch.cuda.manual_seed_all(args.seed)  # 设置所有GPU的随机种子

    torch.backends.cudnn.benchmark = True  # 为True时，每次返回的卷积算法将是确定的，即默认算法。如果设置为False，那么每次卷积算法都将从所有可用的卷积算法中进行选择，这样就会增加计算时间，但可以
    
    return args, new_config

def main():
    args, config  = parse_args_and_config()  # 
    logging.info(f"Writing log file to {args.log_path}")
    logging.info(f"Exp instance ID = {os.getpid()}")
    logging.info(f"Exp comment = {args.comment}")
    try:
        runner = GAN(args, config)
        if args.sample:  # 
            runner.sample()
        else:
            runner.train()
    except Exception:
        logging.error(traceback.format_exc())


if __name__ == '__main__':
    sys.exit(main())  
