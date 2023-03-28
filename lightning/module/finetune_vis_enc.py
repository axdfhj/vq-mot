import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from VQ_utils.CLIP_encoder import CLIPTextEmbedding, CLIPImageEmbedding
import wandb
from VQ_utils.clip_grad_norm import ClipGradNorm
import pdb
import os

class finetune_visual(pl.LightningModule):
    def __init__(
        self, args, logger) -> None:
        super().__init__()
        self.args = args
        self.visual_enc = CLIPImageEmbedding().cuda()
        self.visual_enc.finetune()
        self.text_enc = CLIPTextEmbedding(pick_last_embedding=True).cuda()
        self.optimizer = torch.optim.AdamW(self.visual_enc.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
        self.save_hyperparameters()
        self.Loss = nn.CrossEntropyLoss()
        self.train_step = 0
        self.automatic_optimization = False
        self.save_path = os.path.join(args.out_dir, 'checkpoints')
        self.info_logger = logger
        self.clip_grad_norm = ClipGradNorm(start_iteration=0, end_iteration=5000, max_norm=0.5)

    def configure_optimizers(self):
        return {"optimizer":self.optimizer}
    
    def manual_backward(self, loss):
        if self.train_step <= self.args.warm_up_iter:
            current_lr = self.args.lr * (self.train_step + 1) / (self.args.warm_up_iter + 1)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
        self.optimizer.zero_grad()
        loss.backward()
        self.clip_grad_norm(self.visual_enc.parameters())
        self.optimizer.step()
        if self.train_step > self.args.warm_up_iter:
            self.scheduler.step()
        self.train_step += 1
        
    def training_step(self, batch, batch_idx):
        images, texts, index = batch['images'].cuda(), batch['texts'], batch['index'].cuda()
        bs = len(texts)
        vis_feats = self.visual_enc.encode_image(images)
        vis_feat = torch.cat([vis_feats[index[i]: index[i+1], :].mean(dim=0, keepdim=True) for i in range(bs)], dim=0)
        vis_feat = vis_feat / vis_feat.norm(dim=-1, keepdim=True)
        text_feat = self.text_enc.encode_text(texts)
        similarity = vis_feat @ text_feat.t()
        exp = (torch.eye(bs) + 1e-10).cuda()
        cal_loss = self.Loss(similarity, exp)
        self.log('train_loss', float(cal_loss))
        if self.train_step % 25 == 0:
            msg = f'step {self.train_step} train_loss : {float(cal_loss)}'
            self.info_logger.info(msg)
        self.manual_backward(cal_loss)
        return cal_loss
    
    def validation_step(self, batch, batch_idx):
        images, texts, index = batch['images'].cuda(), batch['texts'], batch['index'].cuda()
        bs = len(texts)
        vis_feats = self.visual_enc.encode_image(images)
        vis_feat = torch.cat([vis_feats[index[i]: index[i+1], :].mean(dim=0, keepdim=True) for i in range(bs)], dim=0)
        vis_feat = vis_feat / vis_feat.norm(dim=-1, keepdim=True)
        text_feat = self.text_enc.encode_text(texts)
        similarity = vis_feat @ text_feat.t()
        exp = (torch.eye(bs) + 1e-10).cuda()
        cal_loss = self.Loss(similarity, exp)
        msg = f'step {self.train_step} val_loss : {float(cal_loss)}'
        self.info_logger.info(msg)
        self.log_dict({'val_loss': float(cal_loss)})
        return cal_loss
    
    def validation_epoch_end(self, outputs):
        if self.local_rank == 0:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save({'net': self.visual_enc.state_dict()}, os.path.join(self.save_path, f'epoch-{self.current_epoch}-step-{self.global_step}.pth'))
    
    def test_step(self, batch, batch_idx):
        return
    
    def forward(self, prompt, length):
        return