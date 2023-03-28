import pdb
import os
from os.path import join as pjoin
import pytorch_lightning as pl
from lightning.module.finetune_blip2 import finetune_blip2
from blip2.dataset.finetune_dataset import TextGenerateDATALoader
import json
import utils.utils_model as utils_model
import warnings
import argparse
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf
warnings.filterwarnings('ignore')

def main():
    
    ##### ---- Exp dirs ---- #####
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-path', default='blip2/config/blip2_inference.yaml', type=str, help='model options')
    parser.add_argument('--dataname', default=None, type=str, help='model options')
    parser.add_argument('--nbeams', default=5, type=int, help='number of beams')
    parser.add_argument('--batchsize', default=8, type=int, help='batch size')
    args_inference = parser.parse_args()
    
    opt = OmegaConf.load(args_inference.config_path)
    finetune_conf = pjoin(os.path.dirname(opt['model']['finetuned_checkpoint']), '..', 'config.yaml')
    args = OmegaConf.load(finetune_conf)
    save_path = pjoin(f'dataset/{args_inference.dataname}', f'texts')
    os.makedirs(save_path, exist_ok=True)

    pl.seed_everything(args.seed)
    ##### ---- model ---- #####
    model = finetune_blip2(args, opt).cuda()
    
    ##### ---- data ---- #####
    dataloader = TextGenerateDATALoader(dataname=args_inference.dataname, batch_size=8, nodebug=True)

    ##### ---- generate ---- #####
    for batch in tqdm(dataloader):
        name_list = batch['name_list']
        texts = model(batch, args_inference.nbeams)
        for i in range(len(name_list)):
            text_dir = pjoin(save_path, name_list[i] + '.txt')
            file = open(text_dir, 'a')
            prompt = f'{texts[i]}\n'
            file.write(prompt)
            file.close()
    
if __name__ == "__main__":
    main()