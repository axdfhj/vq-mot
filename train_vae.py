import os 
import pdb
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.module.vq_vae import vq_vae
from lightning.datamodule.humanml3d_vae import Humanml3dVaeDataModule
import json
import options.option_vq as option_trans
import utils.utils_model as utils_model
import warnings
import yaml
warnings.filterwarnings('ignore')

def main():
    
    ##### ---- Exp dirs ---- #####
    args = option_trans.get_args_parser()    
    pl.seed_everything(args.seed)

    if not args.nodebug:
        args.exp_name = 'db-' + args.exp_name
        os.makedirs(os.path.join(args.out_dir, 'debug'), exist_ok=True)
        args.out_dir = os.path.join(args.out_dir, 'debug', f'{args.exp_name}')
    else:
        os.makedirs(os.path.join(args.out_dir, 'vq-vae'), exist_ok=True)
        args.out_dir = os.path.join(args.out_dir, 'vq-vae', f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok = True)
    
    with open(os.path.join(args.out_dir, 'config.yaml'), 'w') as file:
        yaml.dump({**vars(args)}, file)
    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    logger.info(json.dumps({**vars(args)}, indent=4, sort_keys=True))
    
    
    wandb_logger = pl_loggers.WandbLogger(
        # set the wandb project where this run will be logged
        project="vqvae_t2m_retrain",
        name=args.exp_name,
        offline= not args.nodebug,
        # track hyperparameters and run metadata
        save_dir=args.out_dir,
        log_model=args.nodebug,
        anonymous=False,
    )
    
    ##### ---- model ---- #####
    model = vq_vae(args)
    
    ##### ---- data ---- #####
    datamodule = Humanml3dVaeDataModule(args)

    ##### ---- trainer ---- #####
    trainer = pl.Trainer(
        benchmark=False,
        max_epochs=2500,
        # max_steps=args.total_iter,
        accelerator="gpu",
        devices=[0, 1, 2, 3] if args.nodebug else [0],
        # strategy='ddp',
        # move_metrics_to_cpu=True,
        default_root_dir=args.out_dir,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=[wandb_logger],
        # val_check_interval=args.eval_iter,
        check_val_every_n_epoch=25,
    )
    
    trainer.fit(model, datamodule=datamodule)
    
if __name__ == "__main__":
    main()