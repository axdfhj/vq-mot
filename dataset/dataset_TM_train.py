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
FLAG3D_TRAIN = [1, 2, 3, 4, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60]
pre_list = ['maybe ', 'as if ', 'looks like ', '']
mid_list = ['', 'the person ', 'the man ', 'a man ', 'the figure ', 'he ', 'a person ']
pose_dict = {'Directions': ['gave directions to someone.', 'make directions.', 'directing.', 'directs for others.'],
             'Discussion': ['is having a discussion.', 'is discussing with somebody.', 'led the discussion.'],
             'Eating': ['eating sandwich.', 'take dinner.', 'eats.', 'is eating something.'],
             'Greeting': ['is greeting.', 'greeted.', 'greeted with other.'],
             'Phoning': ['is calling.', 'make a phone call.', 'phoning.', 'is on the phone talking.'],
             'Posing': ['make a pose.', 'posing.', 'pose for a painting.'],
             'Photo': ['make a photo.', 'pose for a photo.'],
             'Purchases': ['are buying something.', 'purchases.'],
             'Sitting': ['is sitting on a chair.'],
             'SittingDown': ['was sitting on the bench.', 'sat down.'],
             'Smoking': ['is smoking.', 'smokes a cigarette.'],
             'TakingPhoto': ['is taking a photo.', 'takes photos for others.', 'take pictures.'],
             'Waiting': [''],
             'Walking': [''],
             'WalkingDog': [''],
             'WalkDog': [''],
             'WalkTogether': ['']}

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
            self.data_root = '/nas/hml3d_datasets/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 't2m_right':
            self.data_root = '/nas/hml3d_datasets/HumanML3D_right'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = '/nas/hml3d_datasets/HumanML3D_right'
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
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 26 if unit_length == 8 else 51
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 't2m_right':
            self.data_root = '/nas/hml3d_datasets/HumanML3D_right'
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
                                while new_name in data_dict.keys():
                                    new_name = '_' + new_name
                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                    'text':[text_dict],
                                                    'loss_weight': 1,
                                                    'source': 'hml_nogtext'}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                    'text':text_data,
                                    'loss_weight': 1,
                                    'source': 'hml_nogtext'}
                    new_name_list.append(name)
            except:
                pass
        print(f'get {len(data_dict)} motions from {self.data_root}!')
        
        #get motion from flag3d
        text_root = '/nas/flag3d/clean_data/texts'
        motion_root = '/nas/flag3d/clean_data'
        
        m_file_list = [glob(pjoin(motion_root, tokenizer_name, f'M_M*P{p+1:03d}A{a+1:03d}R*.npy')) for a in FLAG3D_TRAIN for p in range(10)]
        file_list = [glob(pjoin(motion_root, tokenizer_name, f'M*P{p+1:03d}A{a+1:03d}R*.npy')) for a in FLAG3D_TRAIN for p in range(10)]
        file_list = [[x for x in ls if not os.path.basename(x).startswith('M_')] for ls in file_list]
        file_list += m_file_list
        num = 0
        for motion_list in tqdm(file_list):
            if len(motion_list) == 0:
                continue
            mname = motion_list[0][-16:-8]
            if os.path.basename(motion_list[0]).startswith('M_'):
                mname = 'M_' + mname
            m_token_list = []
            text_data = []
            for path in motion_list:
                code = np.load(path)
                if True in np.isnan(code):
                    continue
                m_token_list.append(code)
                name = path[-20:-4]
                try:
                    with cs.open(pjoin(text_root, name + '.txt')) as f:
                        lines = f.readlines()
                        for line in lines[0:4]:
                            text_dict = {}
                            caption = line.strip()
                            text_dict['caption'] = caption
                            text_data.append(text_dict)
                except:
                    pass
            data_dict[mname] = {'m_token_list': m_token_list,
                                       'text':text_data,
                                       'loss_weight': 1,
                                       'source': 'flag'}
            new_name_list.append(mname)
            if len(data_dict) != len(new_name_list):
                pdb.set_trace()
            num += 1
        print(f'get {num} motions from {motion_root}!')
        print(f'dataloader for fintune!')
        # get motion from h36m
        text_root = '/nas/hml3d_datasets/h36m/texts'
        motion_root = pjoin('/nas/hml3d_datasets/h36m', tokenizer_name)
        
        num = 0
        for p in tqdm(os.listdir(motion_root)):
            motion_path = pjoin(motion_root, p)
            m_token_list = np.load(motion_path)
            if True in np.isnan(m_token_list):
                print(p)
                continue
            mname = p[:-4]
            text_data = []
            try:
                textname = mname if not mname.startswith('M') else mname[1:]
                with cs.open(pjoin(text_root, textname + '_mesh_frames.txt')) as f:
                    lines = f.readlines()
                    for line in lines:
                        text_dict = {}
                        caption = line.strip()
                        text_dict['caption'] = caption
                        text_data.append(text_dict)
            except:
                print(pjoin(text_root, mname + '_mesh_frames.txt'))
                continue
            action = textname.split('_')[1]
            data_dict[mname] = {'m_token_list': m_token_list,
                                       'text':text_data,
                                       'action': action,
                                       'loss_weight': 1,
                                       'source': 'h36m'}
            new_name_list.append(mname)
            num += 1
            if len(data_dict) != len(new_name_list):
                pdb.set_trace()
        print(f'get {num} motions from {motion_root}!')
        assert len(data_dict) == len(new_name_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        loss_weight = data['loss_weight']
        m_token_list, text_list = data['m_token_list'], data['text']
        if data['source'] == 'flag':
            m_token_list = random.choice(m_token_list)
        m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        caption= text_data['caption']
        if data['source'] == 'h36m':
            action = data['action']
            action_sentence = random.choice(pre_list) + random.choice(mid_list) + random.choice(pose_dict[action])
            coin = np.random.choice([1, 2, 3, 4])
            if coin == 1:
                caption = caption + ' ' + action_sentence
            elif coin == 2:
                caption = action_sentence + caption
            elif coin == 3:
                caption = action_sentence
            else:
                caption = caption

        if len(m_tokens) > self.max_motion_length:
            idx = random.randint(0, len(m_tokens) - self.max_motion_length)
            randlen = random.randint(self.max_motion_length-11, self.max_motion_length-1)
            m_tokens = m_tokens[idx: idx+randlen]
        
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
        # guidance_free
        guidance_free_prob = 0.05 if data['source'] in ['hml_nogtext', 'hml'] else 0.95
        if random.random() < guidance_free_prob:
            caption = ''
        return caption, m_tokens.reshape(-1), m_tokens_len, loss_weight

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
    num_workers = num_workers if nodebug else 0
    train_loader = torch.utils.data.DataLoader(Text2MotionDataset_withH36m(dataset_name, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length, nodebug=nodebug),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                            #   collate_fn=collate_fn,
                                              drop_last = True)

    return train_loader