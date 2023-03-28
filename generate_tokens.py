import models.vqvae as vqvae
from os.path import join as pjoin
import options.option_transformer as option_trans
from tqdm import tqdm
import os 
import torch
import numpy as np
import pdb
from dataset import dataset_tokenize

args = option_trans.get_args_parser()
torch.manual_seed(args.seed)
import warnings
warnings.filterwarnings('ignore')

data_roots = {'h36m': '/nas/hml3d_datasets/h36m', 't2m': '/nas/hml3d_datasets/HumanML3D', 't2m_right': '/nas/hml3d_datasets/HumanML3D_right'}

train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t)

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

for batch in tqdm(train_loader_token):
    pose, name = batch
    bs, seq = pose.shape[0], pose.shape[1]

    pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
    target = net.encode(pose)
    target = target.cpu().numpy()
    np.save(pjoin(data_roots[args.dataname], args.vq_dir, name[0] +'.npy'), target)