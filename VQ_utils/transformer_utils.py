# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import math
import torch
from torch import nn
import torch.nn.functional as F
import pdb

import numpy as np
from einops import rearrange
from inspect import isfunction
from torch.cuda.amp import autocast
from torch.utils.checkpoint import checkpoint

class BaseEmbedding(nn.Module):

    def get_loss(self):
        return None

    def forward(self, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        if self.trainable and mode:
            super().train()
        return self

    def _set_trainable(self):
        if not self.trainable:
            for pn, p in self.named_parameters():
                p.requires_grad = False
            self.eval()   

class D2MMaskMusicEmbedding(BaseEmbedding):
    def __init__(self,
                 num_embed=2048,
                 spatial_size=[1, 16537], # height and with 
                 embed_dim=3968, 
                 trainable=True,
                 pos_emb_type='embedding'

        ):
        super().__init__()
        
        if isinstance(spatial_size, int):
            spatial_size = [spatial_size, spatial_size]

        self.spatial_size = spatial_size
        self.num_embed = num_embed + 1
        self.embed_dim = embed_dim
        self.trainable = trainable
        self.pos_emb_type = pos_emb_type

        assert self.pos_emb_type in ['embedding', 'parameter']
        
        self.emb = nn.Embedding(self.num_embed, embed_dim)
        if self.pos_emb_type == 'embedding':
            self.height_emb = nn.Embedding(self.spatial_size[0], embed_dim) # height   
            self.width_emb = nn.Embedding(self.spatial_size[1], embed_dim) # width
        else:
            self.height_emb = nn.Parameter(torch.zeros(1, self.spatial_size[0], embed_dim)) # height #32,1024
            self.width_emb = nn.Parameter(torch.zeros(1, self.spatial_size[1], embed_dim)) # width   #32,1024
        
        self._set_trainable()

    def forward(self, index, **kwargs):
        assert index.dim() == 2 # B x L
        try:
            index[index < 0] = 0  
            emb = self.emb(index)
        except:
            raise RuntimeError('IndexError: index out of range in self, max index {}, num embed {}'.format(index.max(), self.num_embed))
        
        # add col and row embedding
        if emb.shape[1] > 0:
        # if False:
            if self.pos_emb_type == 'embedding':
                height_emb = self.height_emb(torch.arange(self.spatial_size[0], device=index.device).view(1, self.spatial_size[0])).unsqueeze(2) # 1 x H x D -> 1 x H x 1 x D
                width_emb = self.width_emb(torch.arange(self.spatial_size[1], device=index.device).view(1, self.spatial_size[1])).unsqueeze(1) # 1 x W x D -> 1 x 1 x W x D
            else:
                height_emb = self.height_emb.unsqueeze(2) # 1 x H x D -> 1 x H x 1 x D
                width_emb = self.width_emb.unsqueeze(1) # 1 x W x D -> 1 x 1 x W x D
            pos_emb = (height_emb + width_emb).view(1, self.spatial_size[0] * self.spatial_size[1], -1) # 1 x H x W x D -> 1 x L xD
            emb = emb + pos_emb[:, :emb.shape[1], :]

        # print("check mask embedding size:", emb.size(), emb)
        # exit()
        return emb

class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 causal=True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
        self.causal = causal

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

class CrossAttention(nn.Module):
    def __init__(self,
                 condition_seq_len,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 seq_len=None, # the max length of sequence
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
                 causal=True,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.causal = causal

        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            self.register_buffer("mask", torch.tril(torch.ones(seq_len, seq_len))
                                        .view(1, 1, seq_len, seq_len))

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att

class GELU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * F.sigmoid(1.702 * x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps, dim, rescale_steps=4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x):
        x = x / self.num_steps * self.rescale_steps
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adalayernorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class AdaInsNorm(nn.Module):
    def __init__(self, n_embd, diffusion_step, emb_type="adainsnorm_abs"):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(diffusion_step, n_embd)
        else:
            self.emb = nn.Embedding(diffusion_step, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd*2)
        self.instancenorm = nn.InstanceNorm1d(n_embd)

    def forward(self, x, timestep):
        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.instancenorm(x.transpose(-1, -2)).transpose(-1,-2) * (1 + scale) + shift
        return x

class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 class_type='adalayernorm',
                 class_number=1000,
                 condition_seq_len=77,
                 n_embd=1024,
                 n_head=16,
                 seq_len=256,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 attn_type='full',
                 if_upsample=False,
                 upsample_type='bilinear',
                 upsample_pre_channel=0,
                 content_spatial_size=None, # H , W
                 conv_attn_kernel_size=None, # only need for dalle_conv attention
                 condition_dim=1024,
                 diffusion_step=100,
                 timestep_type='adalayernorm',
                 window_size = 8,
                 mlp_type = 'fc',
                 ):
        super().__init__()
        self.if_upsample = if_upsample
        self.attn_type = attn_type

        if attn_type in ['selfcross', 'selfcondition', 'self']: 
            if 'adalayernorm' in timestep_type:
                self.ln1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")
        else:
            self.ln1 = nn.LayerNorm(n_embd)
        
        self.ln2 = nn.LayerNorm(n_embd)
        # self.if_selfcross = False
        if attn_type in ['self', 'selfcondition']:
            self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                seq_len=seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )
            if attn_type == 'selfcondition':
                if 'adalayernorm' in class_type:
                    self.ln2 = AdaLayerNorm(n_embd, class_number, class_type)
                else:
                    self.ln2 = AdaInsNorm(n_embd, class_number, class_type)
        elif attn_type == 'selfcross':
            self.attn1 = FullAttention(
                    n_embd=n_embd,
                    n_head=n_head,
                    seq_len=seq_len,
                    attn_pdrop=attn_pdrop, 
                    resid_pdrop=resid_pdrop,
                    )
            self.attn2 = CrossAttention(
                    condition_seq_len,
                    n_embd=n_embd,
                    condition_embd=condition_dim,
                    n_head=n_head,
                    seq_len=seq_len,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    )
            if 'adalayernorm' in timestep_type:
                self.ln1_1 = AdaLayerNorm(n_embd, diffusion_step, timestep_type)
            else:
                print("timestep_type wrong")
        else:
            print("attn_type error")
        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()
        if mlp_type == 'conv_mlp':
            self.mlp = Conv_MLP(n_embd, mlp_hidden_times, act, resid_pdrop)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )

    def forward(self, x, encoder_output, timestep, mask=None):    
        if self.attn_type == "selfcross":
            a, att = self.attn1(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a
            a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
            x = x + a
        elif self.attn_type == "selfcondition":
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a
            x = x + self.mlp(self.ln2(x, encoder_output.long()))   # only one really use encoder_output
            return x, att
        else:  # 'self'
            a, att = self.attn(self.ln1(x, timestep), encoder_output, mask=mask)
            x = x + a 

        x = x + self.mlp(self.ln2(x))

        return x, att

class Conv_MLP(nn.Module):
    def __init__(self, n_embd, mlp_hidden_times, act, resid_pdrop):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=n_embd, out_channels=int(mlp_hidden_times * n_embd), kernel_size=3, stride=1, padding=1)
        self.act = act
        self.conv2 = nn.Conv2d(in_channels=int(mlp_hidden_times * n_embd), out_channels=n_embd, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        n =  x.size()[1]
        x = rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(n)))
        x = self.conv2(self.act(self.conv1(x)))
        x = rearrange(x, 'b c h w -> b (h w) c')
        return self.dropout(x)

class Text2ImageTransformer(nn.Module):
    def __init__(
        self,
        condition_seq_len=77,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        content_seq_len=1024,
        attn_pdrop=0,
        resid_pdrop=0,
        mlp_hidden_times=4,
        block_activate=None,
        attn_type='selfcross',
        content_spatial_size=[32,32], # H , W
        condition_dim=512,
        diffusion_step=1000,
        timestep_type='adalayernorm',
        content_emb_config=None,
        mlp_type='fc',
        checkpoint=False,
        nb_code=512,# axdfhj maybe 513
    ):
        super().__init__()

        self.use_checkpoint = checkpoint
        # self.content_emb = nn.Embedding(nb_code + 1, n_embd)
        self.content_emb = D2MMaskMusicEmbedding(nb_code + 1, embed_dim=n_embd, spatial_size=content_spatial_size)
        # transformer
        assert attn_type == 'selfcross'
        all_attn_type = [attn_type] * n_layer
        
        if content_spatial_size is None:
            s = int(math.sqrt(content_seq_len))
            assert s * s == content_seq_len
            content_spatial_size = (s, s)

        self.blocks = nn.Sequential(*[Block(
                condition_seq_len,
                n_embd=n_embd,
                n_head=n_head,
                seq_len=content_seq_len,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                attn_type=all_attn_type[n],
                content_spatial_size=content_spatial_size, # H , W
                condition_dim = condition_dim,
                diffusion_step = diffusion_step,
                timestep_type = timestep_type,
                mlp_type = mlp_type,
        ) for n in range(n_layer)])

        # final prediction head
        self.to_logits = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, nb_code),
        )
        
        self.condition_seq_len = condition_seq_len
        self.content_seq_len = content_seq_len

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.elementwise_affine == True:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['condition_emb', 'content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}# if p.requires_grad} 
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def forward(
            self, 
            input, 
            cond_emb,
            t,
            length):
        B, L = input.shape
        cont_emb = self.content_emb(input)
        cum = torch.ones((B, L, 1)).cumsum(dim=1).cuda()
        if isinstance(length, list):
            length = torch.tensor(length)
        length = length.contiguous().view((B,1,1)).cuda()
        mask = (cum <= length)
        emb = cont_emb * mask

        for block_idx in range(len(self.blocks)):   
            if self.use_checkpoint == False:
                emb, att_weight = self.blocks[block_idx](emb, cond_emb, t.cuda()) # B x (Ld+Lt) x D, B x (Ld+Lt) x (Ld+Lt)
            else:
                emb, att_weight = checkpoint(self.blocks[block_idx], emb, cond_emb, t.cuda())
        logits = self.to_logits(emb) # B x (Ld+Lt) x n
        out = rearrange(logits, 'b l c -> b c l')
        return out