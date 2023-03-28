import pdb
import os
from os.path import join as pjoin
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from lightning.module.finetune_blip2 import finetune_blip2
from lightning.datamodule.finetune_data import FinetuneDataModule
import json
import utils.utils_model as utils_model
import warnings
import argparse
import yaml
from omegaconf import OmegaConf
warnings.filterwarnings('ignore')

def main():
    
    ##### ---- Exp dirs ---- #####
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for Amass',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config-path', default='blip2/config/blip2_inference.yaml', type=str, help='model options')
    parser.add_argument('--dataname', default=None, type=str, help='model options')
    args_inference = parser.parse_args()
    
    opt = OmegaConf.load(args_inference.config_path)
    finetune_conf = pjoin(os.path.dirname(opt['model']['finetuned_checkpoint']), '..', 'config.yaml')
    args = OmegaConf.load(finetune_conf)

    pl.seed_everything(args.seed)    
    ##### ---- model ---- #####
    model = finetune_blip2(args, opt)
    
    ##### ---- data ---- #####
    dataname = args_inference.dataname
    datamodule = FinetuneDataModule(args, dataname)

    ##### ---- trainer ---- #####
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=args.epoch,
        accelerator="gpu",
        devices=[0],
        default_root_dir=args.out_dir,
        log_every_n_steps=args.val_every_epoch,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        # callbacks=checkpoint_callback,
        # num_sanity_val_steps=2 if args.nodebug else 0,
    )
    trainer.test(model=model, datamodule=datamodule)
    
if __name__ == "__main__":
    main()