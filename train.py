import os 
import pdb
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.module.vq_mld import vq_diffusion
from lightning.datamodule.humanml3d import Humanml3dDataModule
import json
import options.option_transformer as option_trans
import utils.utils_model as utils_model
import warnings
import yaml
warnings.filterwarnings('ignore')

def main():
    
    ##### ---- Exp dirs ---- #####
    args = option_trans.get_args_parser()
    f = open('VQ_utils/transformer_config.yaml', 'r', encoding='utf-8')
    trans_config = yaml.load(f.read())
    f = open('VQ_utils/scheduler_config.yaml', 'r', encoding='utf-8')
    scheduler_config = yaml.load(f.read())
    #  save hyper
    
    pl.seed_everything(args.seed)
    if not args.nodebug:
        args.exp_name = 'db-' + args.exp_name
        os.makedirs(os.path.join(args.out_dir, 'debug'), exist_ok=True)
        args.out_dir = os.path.join(args.out_dir, 'debug', f'{args.exp_name}')
    else:
        os.makedirs(os.path.join(args.out_dir, 'diffusion'), exist_ok=True)
        args.out_dir = os.path.join(args.out_dir, 'diffusion', f'{args.exp_name}')

    args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
    os.makedirs(args.out_dir, exist_ok = True)
    os.makedirs(args.vq_dir, exist_ok = True)
    
    with open(os.path.join(args.out_dir, 'config.yaml'), 'w') as file:
        yaml.dump({**vars(args), 'trans_config': trans_config, 'scheduler_config': scheduler_config}, file)
    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    logger.info(json.dumps({**vars(args), **trans_config, **scheduler_config}, indent=4, sort_keys=True))
    
    
    wandb_logger = pl_loggers.WandbLogger(
        # set the wandb project where this run will be logged
        project="motion_vq_diffusion",
        name=args.exp_name,
        offline=not args.nodebug,
        # offline=False,
        # track hyperparameters and run metadata
        save_dir=args.out_dir,
        log_model=args.nodebug,
        anonymous=False,
    )
    
    ##### ---- model ---- #####
    model = vq_diffusion(args, trans_config, scheduler_config, logger=logger)
    
    ##### ---- data ---- #####
    datamodule = Humanml3dDataModule(args)

    ##### ---- trainer ---- #####
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=args.epoch,
        accelerator="gpu",
        devices=[0, 1, 2, 3] if args.nodebug else [0],
        # devices=[0],
        strategy="ddp",
        # move_metrics_to_cpu=True,
        default_root_dir=args.out_dir,
        log_every_n_steps=args.val_every_epoch,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=[wandb_logger],
        # callbacks=checkpoint_callback,
        # check_val_every_n_epoch=args.val_every_epoch,
        check_val_every_n_epoch=args.val_every_epoch,
        num_sanity_val_steps=2,
    )
    
    trainer.fit(model, datamodule=datamodule)
    
if __name__ == "__main__":
    main()