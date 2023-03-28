import pdb
import os
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from lightning.module.finetune_blip2 import finetune_blip2
from lightning.datamodule.finetune_data import FinetuneDataModule
import json
import options.option_finetune as option_finetune
import utils.utils_model as utils_model
import warnings
import yaml
from omegaconf import OmegaConf
warnings.filterwarnings('ignore')

def main():
    
    ##### ---- Exp dirs ---- #####
    args = option_finetune.get_args_parser()
    opt = OmegaConf.load(args.config_path)
    #  save hyper
    
    pl.seed_everything(args.seed)
    args.exp_name = 'ft-' + args.exp_name
    if not args.nodebug:
        args.exp_name = 'db-' + args.exp_name
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok = True)
    
    with open(os.path.join(args.out_dir, 'config.yaml'), 'w') as file:
        yaml.dump({**vars(args)}, file)
    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    # logger.info(json.dumps({**vars(args)}, indent=4, sort_keys=True))
    
    
    wandb_logger = pl_loggers.WandbLogger(
        # set the wandb project where this run will be logged
        project="finetune_blip2",
        name=args.exp_name,
        offline= not args.nodebug,
        # offline= False,
        # track hyperparameters and run metadata
        save_dir=args.out_dir,
        log_model=args.nodebug,
        anonymous=False,
    )
    
    ##### ---- model ---- #####
    model = finetune_blip2(args, opt, logger)
    
    ##### ---- data ---- #####
    datamodule = FinetuneDataModule(args)

    ##### ---- trainer ---- #####
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=args.epoch,
        accelerator="gpu",
        devices=[0, 1, 2, 3] if args.nodebug else [0],
        # devices=[0],
        strategy="ddp",
        move_metrics_to_cpu=True,
        default_root_dir=args.out_dir,
        log_every_n_steps=args.val_every_epoch,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=[wandb_logger],
        # callbacks=checkpoint_callback,
        check_val_every_n_epoch=args.val_every_epoch,
        # num_sanity_val_steps=2 if args.nodebug else 0,
    )
    
    trainer.fit(model, datamodule=datamodule)
    
if __name__ == "__main__":
    main()