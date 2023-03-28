import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
from VQ_utils.clip_grad_norm import ClipGradNorm
from blip2.model.blip2_opt_arch import Blip2OPT
import pdb
from os.path import join as pjoin
import os

class finetune_blip2(pl.LightningModule):
    def __init__(
        self, args, opt, logger=None) -> None:
        super().__init__()
        self.args = args
        self.model = Blip2OPT(opt['model'])
        self.num_beams = opt['model'].get("num_beams", 5)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
        self.save_hyperparameters()
        self.train_step = 0
        self.automatic_optimization = False
        self.save_path = pjoin(args.out_dir, 'checkpoints')
        self.eval_path = pjoin(args.out_dir, 'eval_texts')
        os.makedirs(self.eval_path, exist_ok=True)
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
        self.clip_grad_norm(self.model.opt_proj.parameters())
        self.optimizer.step()
        if self.train_step > self.args.warm_up_iter:
            self.scheduler.step()
        self.train_step += 1
        
    def training_step(self, batch, batch_idx):
        loss = self.model(batch, train=True)
        self.log('train_loss', float(loss))
        if self.train_step % 50 == 0 and self.info_logger != None:
            msg = f'step {self.train_step} train_loss : {float(loss)}'
            self.info_logger.info(msg)
        self.manual_backward(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        name_list = batch['name_list']
        gt_text = batch['text_input']
        text_list = self.model(batch, train=False)
        text_file = pjoin(self.eval_path, f'epoch:{self.current_epoch}.txt')
        file = open(text_file, 'a')
        for i in range(len(name_list)):
            if name_list[i].startswith('M'):
                name = name_list[i][1:]
            else:
                name = name_list[i]
            msg = f'{name[:6]}: {text_list[i]} \ {gt_text[i]}\n'
            file.write(msg)
        file.close()
    
    def validation_epoch_end(self, outputs):
        if self.local_rank == 0 and self.current_epoch > 0:
            os.makedirs(self.save_path, exist_ok=True)
            torch.save({'model': self.model.state_dict()}, pjoin(self.save_path, f'epoch-{self.current_epoch}-step-{self.global_step}.pth'))
    
    def test_step(self, batch, batch_idx):
        name_list = batch['name_list']
        gt_text = batch['text_input']
        text_list = self.model(batch, train=False)
        text_file = pjoin(self.eval_path, f't-epoch:{self.current_epoch}-nbeam:{self.num_beams}.txt')
        file = open(text_file, 'a')
        for i in range(len(name_list)):
            if name_list[i].startswith('M'):
                name = name_list[i][1:]
            else:
                name = name_list[i]
            msg = f'{name[:6]}: {text_list[i]} \ {gt_text[i]}\n'
            file.write(msg)
        file.close()
        return
    
    def forward(self, samples, nbeams = 1):
        output_text = self.model.generate(samples, num_beams=nbeams)
        return output_text