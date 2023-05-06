import os 
import pdb
from os.path import join as pjoin
import numpy as np
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from lightning.module.vq_mld import vq_diffusion
from lightning.datamodule.humanml3d import Humanml3dDataModule
import utils.utils_model as utils_model
import shutil
import warnings
import argparse
from omegaconf import OmegaConf
warnings.filterwarnings('ignore')

def main():
    ##### ---- Exp dirs ---- #####
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint_dir', type=str, default='', help='dir of checkpoint')
    parser.add_argument('--replication_times', type=int, default=3, help='times of test')
    parser.add_argument('--output_dir', type=str, default='.', help='output dir')
    opt = parser.parse_args()
    
    ##### ---- get config ---- #####
    config_path = pjoin(os.path.dirname(opt.checkpoint_dir), '..', 'config.yaml')
    args = OmegaConf.load(config_path)
    args.out_dir = os.path.join(args.out_dir, 'Test-' + os.path.basename(opt.checkpoint_dir)[:-4])
    os.makedirs(opt.output_dir, exist_ok=True)
    args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
    args.resume_trans = opt.checkpoint_dir
    os.makedirs(args.out_dir, exist_ok = True)
    trans_config = args.trans_config
    scheduler_config = args.scheduler_config
    # pl.seed_everything(args.seed)
    ##### ---- log ---- #####    
    logger = utils_model.get_logger(args.out_dir)
    
    ##### ---- model ---- #####
    model = vq_diffusion(args, trans_config, scheduler_config, logger=logger).cuda()
    
    ##### ---- get input ---- #####
    input_file = 'demo.txt'
    shutil.copy(input_file, opt.output_dir)
    prompts = []
    lengths = []
    with open(input_file, 'r') as file:
        line = file.readline()
        while line:
            prompt, length = line.strip().split('#')
            prompts.append(prompt)
            lengths.append(int(length))
            line = file.readline()
    
    ##### ---- generate ---- #####
    model(prompts, lengths, sample_times=opt.replication_times, output_root=opt.output_dir)
    
if __name__ == "__main__":
    main()
