import os 
import pdb
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from vq_vae import vq_vae
from humanml3d_vae import Humanml3dVaeDataModule
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

    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok = True)
    
    with open(os.path.join(args.out_dir, 'config.yaml'), 'w') as file:
        yaml.dump({**vars(args)}, file)
    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    logger.info(json.dumps({**vars(args)}, indent=4, sort_keys=True))
    
    
    wandb_logger = pl_loggers.WandbLogger(
        # set the wandb project where this run will be logged
        project="motion_vq_vae",
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
        max_steps=args.total_iter,
        accelerator="gpu",
        devices=[0, 1, 2, 3] if args.nodebug else [0],
        # devices=[0],
        strategy="ddp",
        move_metrics_to_cpu=True,
        default_root_dir=args.out_dir,
        log_every_n_steps=args.eval_iter,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=[wandb_logger],
        # callbacks=checkpoint_callback,
        check_val_every_n_epoch=args.eval_iter,
    )
    
    trainer.fit(model, datamodule=datamodule)
    
if __name__ == "__main__":
    main()