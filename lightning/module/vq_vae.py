import pytorch_lightning as pl
import numpy as np
import torch
import clip
import models.vqvae as vqvae
from options.get_eval_option import get_opt
import utils.utils_model as utils_model
import utils.losses as losses
from models.evaluator_wrapper import EvaluatorModelWrapper
from VQ_utils.clip_grad_norm import ClipGradNorm
from utils.eval_trans import calculate_R_precision, calculate_activation_statistics, calculate_diversity, calculate_frechet_distance, calculate_multimodality
from utils.motion_process import recover_from_ric
import pdb
import os
from os.path import join as pjoin

class vq_vae(pl.LightningModule):
    def __init__(
        self,
        args, train=True) -> None:
        
        super().__init__()
        self.args = args
        dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        wrapper_opt = get_opt(dataset_opt_path, args.nodebug)
        self.eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta' if args.dataname == 'kit' else 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        if args.dataname == 't2m':
            self.mean = np.load(pjoin('dataset/HumanML3D', 'Mean.npy'))
            self.std = np.load(pjoin('dataset/HumanML3D', 'Std.npy'))
        elif args.dataname == 't2m_right':
            self.mean = np.load(pjoin('dataset/HumanML3D_right', 'Mean.npy'))
            self.std = np.load(pjoin('dataset/HumanML3D_right', 'Std.npy'))
        else:
            self.mean = np.load(pjoin(meta_dir, 'mean.npy'))
            self.std = np.load(pjoin(meta_dir, 'std.npy'))
        self.net = vqvae.HumanVQVAE(args)
        self.save_path = os.path.join(args.out_dir, 'checkpoints')
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
        self.Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)
        if train:
            self.save_hyperparameters()
            self.evaluation_epoch_init()
        self.train_step = 0
        self.automatic_optimization = False
        self.bestfid = 0.15
    
    def load_checkpoint(self, path):
        state_dict = torch.load(path, map_location="cpu")
        self.net.load_state_dict(state_dict['net'], strict=False)
        
    def evaluation_epoch_init(self):
        self.motion_annotation_list = []
        self.motion_pred_list = []
        self.R_precision_real = 0
        self.R_precision = 0
        self.matching_score_real = 0
        self.matching_score_pred = 0
        self.nb_sample = 0
        self.motion_multimodality = []
        
    def configure_optimizers(self):
        return {"optimizer":self.optimizer}
    
    def manual_backward(self, loss):
        if self.train_step <= self.args.warm_up_iter:
            current_lr = self.args.lr * (self.train_step + 1) / (self.args.warm_up_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.train_step > self.args.warm_up_iter:
            self.scheduler.step()
        self.train_step += 1
        
    def training_step(self, batch, batch_idx):
        gt_motion = batch.cuda().float()
        input_motion = gt_motion
        pred_motion, loss_commit, perplexity = self.net(input_motion)
        loss_motion = self.Loss(pred_motion, gt_motion)
        loss_vel = self.Loss.forward_vel(pred_motion, gt_motion)
        
        loss = loss_motion + self.args.commit * loss_commit + self.args.loss_vel * loss_vel
        self.log("train_loss", loss)
        self.manual_backward(loss)
        return loss
    
    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def evaluation_step(self, batch, mm=False):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
        et, em = self.eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]
        num_joints = 21 if motion.shape[-1] == 251 else 22
        motion_multimodality_batch = []
        sample_times = 30 if mm else 1
        for i in range(sample_times):
            pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()
            for k in range(bs):
                pose = self.inv_transform(motion[k:k+1, :m_length[k], :].detach().cpu().numpy())
                pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                pred_pose, loss_commit, perplexity = self.net(motion[k:k+1, :m_length[k]])
                pred_denorm = self.inv_transform(pred_pose.detach().cpu().numpy())
                pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
                
                pred_pose_eval[k:k+1,:m_length[k],:] = pred_pose


            et_pred, em_pred = self.eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            if i == 0:
                self.motion_annotation_list.append(em)
                self.motion_pred_list.append(em_pred)

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                self.R_precision_real += temp_R
                self.matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                self.R_precision += temp_R
                self.matching_score_pred += temp_match
                self.nb_sample += bs
        if mm:
            self.motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))
    
    def evaluation_epoch_end(self, mm=False):
        motion_annotation_np = torch.cat(self.motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(self.motion_pred_list, dim=0).cpu().numpy()
        gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
        mu, cov= calculate_activation_statistics(motion_pred_np)

        diversity_real = calculate_diversity(motion_annotation_np, 300 if self.nb_sample > 300 else 63)
        diversity = calculate_diversity(motion_pred_np, 300 if self.nb_sample > 300 else 63)

        R_precision_real = self.R_precision_real / self.nb_sample
        R_precision = self.R_precision / self.nb_sample

        matching_score_real = self.matching_score_real / self.nb_sample
        matching_score_pred = self.matching_score_pred / self.nb_sample
        
        multimodality=0
        if mm:
            motion_multimodality = torch.cat(self.motion_multimodality, dim=0).cpu().numpy()
            multimodality = calculate_multimodality(motion_multimodality, 10)
        
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        if mm:
            self.log_dict({"fid": fid, "matching_score": matching_score_pred, "r1": R_precision[0], "r2": R_precision[1], "r3": R_precision[2], "diversity": diversity, "multimodality": multimodality})
        else:
            self.log_dict({"fid": fid, "matching_score": matching_score_pred, "r1": R_precision[0], "r2": R_precision[1], "r3": R_precision[2], "diversity": diversity})
        self.evaluation_epoch_init()
        return fid, diversity, R_precision, matching_score_pred, multimodality
    
    def validation_step(self, batch, batch_idx):
        self.evaluation_step(batch, mm=False)

    def validation_epoch_end(self, outputs):
        fid, diversity, R_precision, matching_score_pred, multimodality = self.evaluation_epoch_end(mm=False)
        if fid < self.bestfid:
            self.bestfid = fid
            if self.local_rank not in [1, 2, 3]:
                os.makedirs(self.save_path, exist_ok=True)
                torch.save({'net': self.net.state_dict()}, os.path.join(self.save_path, f'best_fid.pth'))
        if fid < 0.15:
            if self.local_rank not in [1, 2, 3]:
                os.makedirs(self.save_path, exist_ok=True)
                torch.save({'net': self.net.state_dict()}, os.path.join(self.save_path, f'last.pth'))
        if fid < 0.15 and self.train_step % 10000 == 0:
            if self.local_rank not in [1, 2, 3]:
                os.makedirs(self.save_path, exist_ok=True)
                torch.save({'net': self.net.state_dict()}, os.path.join(self.save_path, f'epoch-{self.current_epoch}-step-{self.train_step}.pth'))
            
    
    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch, mm=False)
        return
    
    def test_epoch_end(self, outputs):
        fid, diversity, R_precision, matching_score_pred, multimodality = self.evaluation_epoch_end(mm=False)
        return