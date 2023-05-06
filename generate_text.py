import pdb
import os
from os.path import join as pjoin
import pytorch_lightning as pl
from lightning.module.finetune_blip2 import finetune_blip2
from blip2.dataset.finetune_dataset import TextGenerateDATALoader, Blip2CheckDATALoader
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
    parser.add_argument('--batchsize', default=16, type=int, help='batch size')
    parser.add_argument('--src-path', default=None, type=str, help='path to frames dir')
    parser.add_argument('--use-nucleus-sampling', action='store_true', help='use nucleus sampling')
    parser.add_argument('--top-p', default=0.9, type=float, help='The cumulative probability for nucleus sampling.')
    args_inference = parser.parse_args()
    
    opt = OmegaConf.load(args_inference.config_path)
    finetune_checkpoint = opt['model'].get("finetuned_checkpoint", None)
    if finetune_checkpoint:
        finetune_conf = pjoin(os.path.dirname(opt['model']['finetuned_checkpoint']), '..', 'config.yaml')
        args = OmegaConf.load(finetune_conf)
    else:
        args = OmegaConf.load('./experiments/ft-blip2_vit_longer/config.yaml')
    if args_inference.dataname:
        save_path = pjoin(f'dataset/{args_inference.dataname}', f'gtexts')
    elif args_inference.src_path:
        save_path = pjoin(args_inference.src_path, '..', f'texts')
    os.makedirs(save_path, exist_ok=True)

    # pl.seed_everything(args.seed)
    ##### ---- model ---- #####
    model = finetune_blip2(args, opt).cuda()
    
    ##### ---- data ---- #####
    # dataloader = TextGenerateDATALoader(dataname=args_inference.dataname, batch_size=8, nodebug=True)
    dataloader = Blip2CheckDATALoader(src_path=args_inference.src_path, batch_size=args_inference.batchsize, nodebug=True)
    text_total = pjoin(save_path, '_total.txt')
    ##### ---- generate ---- #####
    for batch in tqdm(dataloader):
        name_list = batch['name_list']
        if not args_inference.use_nucleus_sampling:
            texts = model(batch, args_inference.nbeams)
        else:
            texts = model(batch, use_nucleus_sampling=True, top_p=args_inference.top_p)
        for i in range(len(name_list)):
            text_dir = pjoin(save_path, name_list[i] + '.txt')
            file = open(text_dir, 'a')
            # prompt = f'{texts[i]}\n'
            if not args_inference.use_nucleus_sampling:
                # prompt = f'{name_list[i]} (epoch23) (1-6) (num-beams {args_inference.nbeams}): {texts[i]}\n'
                prompt = f'{texts[i]}\n'
            else:
                # prompt = f'{name_list[i]} (epoch23) (1-6) (top-p {args_inference.top_p}): {texts[i]}\n'
                prompt = f'{texts[i]}\n'
            file.write(prompt)
            file.close()
            
            file = open(text_total, 'a')
            if not args_inference.use_nucleus_sampling:
                prompt = f'{name_list[i]} (num-beams {args_inference.nbeams}): {texts[i]}\n'
                # prompt = f'{texts[i]}\n'
            else:
                prompt = f'{name_list[i]} (top-p {args_inference.top_p}): {texts[i]}\n'
                # prompt = f'{texts[i]}\n'
            file.write(prompt)
            file.close()
    
if __name__ == "__main__":
    main()