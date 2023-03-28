import pytorch_lightning as pl
import torch
from VQ_utils.CLIP_encoder import CLIPTextEmbedding, CLIPImageEmbedding
import models.vqvae as vqvae
import math
from options.get_eval_option import get_opt
import utils.utils_model as utils_model
from models.evaluator_wrapper import EvaluatorModelWrapper
import yaml
from VQ_utils.diffusion_transformer import DiffusionTransformer
from VQ_utils.lr_scheduler import ReduceLROnPlateauWithWarmup
from VQ_utils.clip_grad_norm import ClipGradNorm
from utils.eval_trans import calculate_R_precision, calculate_activation_statistics, calculate_diversity, calculate_frechet_distance, calculate_multimodality
import pdb
import os

class vq_diffusion(pl.LightningModule):
    def __init__(
        self,
        args, trans_config, scheduler_config) -> None:
        
        super().__init__()
        dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

        self.clip_model = CLIPTextEmbedding(normalize=True)
        wrapper_opt = get_opt(dataset_opt_path, args.nodebug)
        self.eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        self.net = vqvae.HumanVQVAE(args)
        trans_config['nb_code'] = args.nb_code # nb_code + 1
        self.denoiser = DiffusionTransformer(**trans_config) #改成 diffusion denoiser
        if args.nodebug:
            print ('loading checkpoint from {}'.format(args.resume_pth))
            ckpt = torch.load(args.resume_pth, map_location='cpu')
            self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        self.net = self.net.cuda()
        
        self.save_path = os.path.join(args.out_dir, 'checkpoints')
        
        if args.resume_trans is not None:
            print ('loading transformer checkpoint from {}'.format(args.resume_trans))
            ckpt = torch.load(args.resume_trans, map_location='cpu')
            self.denoiser.load_state_dict(ckpt['net'], strict=True)
        self.denoiser = self.denoiser.cuda()
        self.optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, self.denoiser, args.optimizer)
        self.scheduler = ReduceLROnPlateauWithWarmup(self.optimizer, **scheduler_config)
        self.clip_grad_norm = ClipGradNorm(start_iteration=0, end_iteration=5000, max_norm=0.5)
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.evaluation_epoch_init()
        
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
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_grad_norm(self.denoiser.parameters())
        self.optimizer.step()
        self.scheduler.step(loss)
        
    def training_step(self, batch, batch_idx):
        clip_text, m_tokens, m_tokens_len = batch
        m_tokens, m_tokens_len = m_tokens.cuda(), m_tokens_len.cuda()
        bs = m_tokens.shape[0]
        target = m_tokens    # (bs, 26)
        target = target.cuda()
        
        feat_clip_text = self.clip_model.encode_text(clip_text).float()
        input_index = target[:,:-1]
        # denoise
        log_model_prob, loss = self.denoiser._train_loss(x=input_index, cond_emb=feat_clip_text, is_train=True, length=m_tokens_len)
        self.log("train_loss", loss)
        self.manual_backward(loss)
        return loss
    
    def evaluation_step(self, batch, mm=False):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        

        feat_clip_text = self.clip_model.encode_text(clip_text).float()
        sample_times = 30 if mm else 1
        motion_multimodality_batch = []
        for i in range(sample_times):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            for k in range(bs):
                # try:
                m_token = int((int((m_length[k] - 2) / 2) - 1) / 2) + 1
                index_motion, logits = self.denoiser.sample_motion_index(feat_clip_text[k:k+1], torch.tensor([m_token]))
                # except:
                #     index_motion = torch.ones(1,1).cuda().long()

                pred_pose = self.net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

            et_pred, em_pred = self.eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            if i == 0:
                pose = pose.cuda().float()
                et, em = self.eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
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

    def on_validation_epoch_end(self):
        fid, diversity, R_precision, matching_score_pred, multimodality = self.evaluation_epoch_end(mm=False)
        if self.local_rank == 0 and fid < 0.5:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save({'net': self.denoiser.state_dict()}, os.path.join(self.save_path, f'epoch-{self.current_epoch}-step-{self.global_step}.pth'))
    
    def test_step(self, batch, batch_idx):
        self.evaluation_step(batch, mm=False)
        return
    
    def on_test_epoch_end(self):
        fid, diversity, R_precision, matching_score_pred, multimodality = self.evaluation_epoch_end(mm=False)
        return
    
    def forward(self, prompt, length):

        return 