import torch
from torch.utils import data
import numpy as np
import math
import os
from os.path import join as pjoin
from PIL import Image
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
from glob import glob
import clip
import pdb

# def collate_fn(batch):
#     batch.sort(key=lambda x: x[3], reverse=True)
#     return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None, nodebug=False):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.mot_mask_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 51
            kinematic_chain = paramUtil.kit_kinematic_chain

        split_file = pjoin(self.data_root, 'train.txt')


        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        if not nodebug:
            id_list = id_list[:300]
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy'%name))

                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text':text_data}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list
        

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        caption= text_data['caption']

        
        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        if m_tokens_len < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.ones((self.max_motion_length-m_tokens_len), dtype=int) * self.mot_mask_idx], axis=0)
        else:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        return caption, m_tokens.reshape(-1), m_tokens_len

def image_feat(path):
    if path.startswith('M'):
        path = path[1:]
    label = path.split('_')[0]
    action = path[len(label) + 1: -(len(path.split('_')[-1]) + 1)]
    split = int(path[:-4].split('_')[-1])
    image_feat_dirs = f'/HOME/lyh/vq-mot/datasets/Human36M/{label}/Feature_b/{action}.*'
    index_clip_dir = f'/HOME/lyh/vq-mot/datasets/Human36M/clip_index_new/{label}_{action}.npy'
    index_clip = np.load(index_clip_dir).tolist()
    start, end = index_clip[split], index_clip[split + 1] 
    image_feat_list = glob(image_feat_dirs)
    image_feat_list = [np.load(path)[np.newaxis, start: end, :] for path in image_feat_list]
    image_feat = np.concatenate(image_feat_list, axis=0)
    image_feat = np.mean(image_feat, axis=0)
    index = np.linspace(0, end - start - 1, 77).astype(int)
    image_feat = image_feat[index, :]
    std = math.sqrt(0.016)
    noise = np.random.randn(*image_feat.shape) * std
    return (image_feat + noise)[np.newaxis, :]

def get_images(name, preprocess):
    if name.startswith('M'):
        name = name[1:]
    image_root_path = f'dataset/h36m/render_frames/{name}_mesh_frames'
    img_files = glob(pjoin(image_root_path, '*'))
    img_list = [preprocess(Image.open(path)).unsqueeze(0) for path in img_files]
    return torch.cat(img_list, dim=0)

class Text2MotionDataset_h36m(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None, nodebug=False):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.mot_mask_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 51
            kinematic_chain = paramUtil.kit_kinematic_chain

        split_file = pjoin(self.data_root, 'train.txt')


        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        if not nodebug:
            id_list = id_list[:100]
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy'%name))

                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'text':[text_dict],
                                                       't': True}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text':text_data,
                                       't': True}
                    new_name_list.append(name)
            except:
                pass
        h36m_motion_list = glob('/HOME/lyh/vq-mot/dataset/HumanML3D/VQVAE_h36m_total/MS*') + glob('/HOME/lyh/vq-mot/dataset/HumanML3D/VQVAE_h36m_total/S*')
        model, preprocess = clip.load('ViT-B/32', device='cpu', jit=False)
        if not nodebug:
            h36m_motion_list = h36m_motion_list[:300]
        
        for path in tqdm(h36m_motion_list):
            name = os.path.basename(path)[:-4]
            text_data = []
            m_token = np.load(path)
            imgs = get_images(name, preprocess)
            if imgs.size == 0:
                continue
            data_dict[name] = {'m_token_list': m_token,
                                       'imgs':imgs,
                                       'img_num':imgs.shape[0],
                                       't': False}
            new_name_list.append(name)
        padding_path = 'dataset/HumanML3D/render_frames/000000_mesh_frames/frame_0000.png'
        self.padding_img = preprocess(Image.open(padding_path)).unsqueeze(0)
        self.data_dict = data_dict
        self.name_list = new_name_list
        

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        if data['t']:
            m_token_list, text_list = data['m_token_list'], data['text']
            text_data = random.choice(text_list)
            caption= text_data['caption']
            imgs = self.padding_img
            num_of_img = 1
        else:
            m_token_list, imgs, num_of_img = data['m_token_list'], data['imgs'], data['img_num']
            caption = 'padding caption'
        m_tokens = random.choice(m_token_list)

        
        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        if m_tokens_len < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.ones((self.max_motion_length-m_tokens_len), dtype=int) * self.mot_mask_idx], axis=0)
        else:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        return caption, m_tokens.reshape(-1), m_tokens_len, num_of_img, imgs, data['t']

def collate_fn(batch):
    caption = [b[0] for b in batch]
    m_tokens = torch.from_numpy(np.concatenate([b[1][np.newaxis] for b in batch]))
    m_tokens_len = torch.tensor(([b[2] for b in batch]))
    index = torch.from_numpy(np.array([0] + [b[3] for b in batch]).cumsum(axis=-1))
    images = torch.cat([i[4] for i in batch], dim=0)
    mask = torch.tensor([b[5] for b in batch])
    return {'caption': caption, 'm_tokens_len': m_tokens_len, 'm_tokens': m_tokens, 'index': index, 'imgs': images, 'mask': mask}

class Text2MotionDataset_withH36m(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None, nodebug=False):
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.mot_mask_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == 't2m':
            self.data_root = '/nas/hml3d_datasets/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = '/nas/hml3d_datasets/HumanML3D/texts'
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 't2m_right':
            self.data_root = '/nas/hml3d_datasets/HumanML3D_right'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = '/HOME/lyh/vq-mot/dataset/HumanML3D/texts'
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 26 if unit_length == 8 else 51
            kinematic_chain = paramUtil.kit_kinematic_chain

        split_file = pjoin(self.data_root, 'train.txt')


        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # if not nodebug:
        #     id_list = id_list[:300]
        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy'%name))

                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text':text_data}
                    new_name_list.append(name)
            except:
                pass
        h36m_name_list = [path[:-4] for path in os.listdir('/nas/hml3d_datasets/h36m/texts')]
        for h36m_name in tqdm(h36m_name_list):
            with open(f'/nas/hml3d_datasets/h36m/texts/{h36m_name}.txt') as file:
                caption = file.readline()[:-1]
            text_dict = {'caption': caption}
            m_token_list = np.load(f'/nas/hml3d_datasets/h36m/{tokenizer_name}/{h36m_name}.npy')
            data_dict[h36m_name] = {'m_token_list': m_token_list, 'text':[text_dict]}
            new_name_list.append(h36m_name)
        self.data_dict = data_dict
        self.name_list = new_name_list
        

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        caption= text_data['caption']

        
        coin = np.random.choice([False, False, True])
        # print(len(m_tokens))
        if coin:
            # drop one token at the head or tail
            coin2 = np.random.choice([True, False])
            if coin2:
                m_tokens = m_tokens[:-1]
            else:
                m_tokens = m_tokens[1:]
        m_tokens_len = m_tokens.shape[0]

        if m_tokens_len < self.max_motion_length:
            m_tokens = np.concatenate([m_tokens, np.ones((self.max_motion_length-m_tokens_len), dtype=int) * self.mot_mask_idx], axis=0)
        else:
            m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)

        return caption, m_tokens.reshape(-1), m_tokens_len

def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, unit_length=4,
                num_workers = 8, nodebug = False) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length, nodebug=nodebug),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                            #   collate_fn=collate_fn,
                                              drop_last = True)

    return train_loader

def withH36mDATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, unit_length=4,
                num_workers = 8, nodebug = False):
    train_loader = torch.utils.data.DataLoader(Text2MotionDataset_withH36m(dataset_name, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length, nodebug=nodebug),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                            #   collate_fn=collate_fn,
                                              drop_last = True)

    return train_loader