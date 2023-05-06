import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from glob import glob
import os
import pdb

class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4, nodebug = True):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.data_root = '/nas/hml3d_datasets/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 't2m_right':
            self.data_root = './dataset/HumanML3D_right'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        
        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        split_file = pjoin(self.data_root, 'train.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        if not nodebug:
            id_list = id_list[:300]
        
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

            
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4,
               nodebug = True):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length, nodebug=nodebug)
    # prob = trainSet.compute_sampling_prob()
    # sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True,
                                              pin_memory=True,
                                              persistent_workers=True)
    
    return train_loader


# add h36m data
class withH36mVQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4, nodebug = True):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        if dataset_name == 't2m':
            self.data_root = '/nas/hml3d_datasets/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 't2m_right':
            self.data_root = '/nas/hml3d_datasets/HumanML3D_right'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        
        joints_num = self.joints_num
        
        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        split_file = pjoin(self.data_root, 'train.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        if not nodebug:
            id_list = id_list[:300]
        
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append([motion])
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.humanml3d_len = len(self.data)
        h36m_roots = ['/nas/hml3d_datasets/h36m/new_joint_vecs', '/nas/rich_toolkit/data/new_joint_vecs']
        for h36m_root in h36m_roots:
            file_list = os.listdir(h36m_root)
            num = 0
            for file_dir in tqdm(file_list):
                try:
                    motion = np.load(pjoin(h36m_root, file_dir))
                    if motion.shape[0] < self.window_size:
                        continue
                    if True in np.isnan(motion):
                        print(f'nan in {file_dir}!')
                        continue
                    self.lengths.append(motion.shape[0] - self.window_size)
                    self.data.append([motion])
                    num += 1
                except:
                    # Some motion may not exist in KIT dataset
                    pass
            print(f'get {num} motions from {h36m_root}!')
        
        flag3d_root = ['/nas/flag3d/clean_data/new_joint_vecs']
        for root in flag3d_root:
            m_file_list = [glob(pjoin(root, f'M_M*P{p+1:03d}A{a+1:03d}R*.npy')) for a in range(60) for p in range(10)]
            file_list = [glob(pjoin(root, f'M*P{p+1:03d}A{a+1:03d}R*.npy')) for a in range(60) for p in range(10)] 
            file_list = [[x for x in ls if not x.startswith('M_')] for ls in file_list] # bug: f'M*P{p+1:03d}A{a+1:03d}R*.npy' contains f'M_M*P{p+1:03d}A{a+1:03d}R*.npy'
            file_list += m_file_list
            num = 0
            for file_dir in tqdm(file_list):
                motion_list = []
                for file in file_dir:
                    try:
                        motion = np.load(pjoin(root, file))
                        if motion.shape[0] < self.window_size:
                            continue
                        if True in np.isnan(motion):
                            print(f'nan in {file}!')
                            continue
                        motion_list.append(motion)
                    except:
                        # Some motion may not exist in KIT dataset
                        pass
                if len(motion_list) == 0:
                    continue
                self.lengths.append(motion_list[0].shape[0] - self.window_size)
                self.data.append(motion_list)
                num += 1
            print(f'get {num} motions from {root}!')

        # self.h36m_mean = np.load('/HOME/lyh/vq-mot/dataset/h36m/Mean.npy')
        # self.h36m_std = np.load('/HOME/lyh/vq-mot/dataset/h36m/Std.npy')
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion_list = self.data[item]
        
        motion = random.choice(motion_list)
        
        idx = random.randint(0, len(motion) - self.window_size)

        motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        # if item < self.humanml3d_len:
        #     mean = self.mean
        #     std = self.std
        # else:
        #     mean = self.h36m_mean
        #     std = self.h36m_std
        mean = self.mean
        std = self.std
        motion = (motion - mean) / std

        return motion

def withH36mDATALoader(dataset_name,
               batch_size,
               num_workers = 4,
               window_size = 64,
               unit_length = 4,
               nodebug = True):
    
    trainSet = withH36mVQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length, nodebug=nodebug)
    # prob = trainSet.compute_sampling_prob()
    # sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    num_workers = num_workers if nodebug else 0
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True,
                                              pin_memory=True if nodebug else False,
                                              persistent_workers=True if nodebug else False
                                              )
    
    return train_loader