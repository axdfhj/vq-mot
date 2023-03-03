import os 
import pdb
from os.path import join as pjoin
import numpy as np
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from vq_mld import vq_diffusion
from humanml3d import Humanml3dDataModule
import utils.utils_model as utils_model
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
    parser.add_argument('--replication_times', type=int, default=20, help='times of test')
    opt = parser.parse_args()
    
    ##### ---- get config ---- #####
    config_path = pjoin(os.path.dirname(opt.checkpoint_dir), '..', 'config.yaml')
    args = OmegaConf.load(config_path)
    args.out_dir = os.path.join(args.out_dir, 'test-' + os.path.basename(opt.checkpoint_dir)[:-4])
    args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{args.vq_name}')
    args.resume_trans = opt.checkpoint_dir
    os.makedirs(args.out_dir, exist_ok = True)
    trans_config = args.trans_config
    scheduler_config = args.scheduler_config
    # pl.seed_everything(args.seed)
    ##### ---- log ---- #####    
    logger = utils_model.get_logger(args.out_dir)
    wandb_logger = pl_loggers.WandbLogger(
        # set the wandb project where this run will be logged
        project="motion_vq_diffusion_test",
        name='test-' + args.exp_name,
        offline= not args.nodebug,
        # track hyperparameters and run metadata
        save_dir=args.out_dir,
        log_model=args.nodebug,
        anonymous=False,
    )
    
    ##### ---- model ---- #####
    model = vq_diffusion(args, trans_config, scheduler_config)
    
    ##### ---- data ---- #####
    datamodule = Humanml3dDataModule(args)
    
    ##### ---- trainer ---- #####
    trainer = pl.Trainer(
        benchmark=False,
        accelerator="gpu",
        devices=[1],
        # devices=[0],
        strategy="ddp",
        move_metrics_to_cpu=True,
        default_root_dir=args.out_dir,
        deterministic=False,
        detect_anomaly=False,
        enable_progress_bar=True,
        logger=[wandb_logger],
    )
    fid_list, diversity_list, r1_list, r2_list, r3_list, matching_score_pred_list, multimodality_list  = [], [], [], [], [], [], []
    for rep in range(opt.replication_times):
        epoch_result = trainer.test(model=model, datamodule=datamodule)[0]
        fid_list.append(epoch_result['fid'])
        diversity_list.append(epoch_result['diversity'])
        r1_list.append(epoch_result['r1'])
        r2_list.append(epoch_result['r2'])
        r3_list.append(epoch_result['r3'])
        matching_score_pred_list.append(epoch_result['matching_score'])
        # multimodality_list.append(epoch_result['multimodality'])
        print(f'rep {rep} result:')
        print('fid: ', fid_list[-1])
        print('diversity: ', diversity_list[-1])
        print('r1: ', r1_list[-1])
        print('r2: ', r2_list[-1])
        print('r3: ', r3_list[-1])
        print('matching_score: ', matching_score_pred_list[-1])
        msg_rep = f"rep {rep} result: FID. {np.mean(fid_list[-1]):.3f}, Diversity. {np.mean(diversity_list[-1]):.3f}, TOP1. {np.mean(r1_list[-1]):.3f}, TOP2. {np.mean(r2_list[-1]):.3f}, TOP3. {np.mean(r3_list[-1]):.3f}, Matching. {np.mean(matching_score_pred_list[-1]):.3f}\n"
        logger.info(msg_rep)

    print('final result:')
    print('fid: ', sum(fid_list)/opt.replication_times)
    print('diversity: ', sum(diversity_list)/opt.replication_times)
    print('r1: ', sum(r1_list)/opt.replication_times)
    print('r2: ', sum(r2_list)/opt.replication_times)
    print('r3: ', sum(r3_list)/opt.replication_times)
    print('matching_score: ', sum(matching_score_pred_list)/opt.replication_times)
    # print('multimodality: ', sum(multimodality_list)/opt.replication_times)
    
    fid = np.array(fid_list)
    div = np.array(diversity_list)
    top1 = np.array(r1_list)
    top2 = np.array(r2_list)
    top3 = np.array(r3_list)
    matching = np.array(matching_score_pred_list)
    # multi = np.array(multimodality_list)
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(opt.replication_times):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(opt.replication_times):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(opt.replication_times):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(opt.replication_times):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(opt.replication_times):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(opt.replication_times):.3f}"
    logger.info(msg_final)
    
if __name__ == "__main__":
    main()