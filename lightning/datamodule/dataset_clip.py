import torch
from torch.utils import data
import numpy as np
import math
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from glob import glob
import clip
from PIL import Image
import pdb

class FinetuneDataset(data.Dataset):
    def __init__(self, split='train', nodebug=True):
        self.device = "cpu"
        model, preprocess = clip.load('ViT-B/32', device=self.device, jit=False)
        split_file = f'dataset/HumanML3D/{split}.txt'
        fps = 20
        self.data_list = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        if not nodebug:
            id_list = id_list[:300]
        elif split == 'train':
            id_list = random.sample(id_list, 5000)
        
        self.data_dict = {}
        self.new_name_list = []
        for name in tqdm(id_list):
            mirror = False
            if name.startswith('M'):
                mirror = True
            frames_path = f'dataset/HumanML3D/render_frames/{name[1:]}_mesh_frames' if mirror else f'dataset/HumanML3D/render_frames/{name}_mesh_frames'
            if os.path.exists(frames_path):
                if len(os.listdir(frames_path)) == 0:
                    continue
                imgs_path = glob(pjoin(frames_path, '*'))
                image_list = []
                text_data = []
                flag = False
                for path in imgs_path:
                    image = Image.open(path)
                    if mirror:
                        image.transpose(Image.FLIP_LEFT_RIGHT)
                    image = preprocess(image).unsqueeze(0)
                    image_list.append(image)
                texts_path = f'dataset/HumanML3D/texts/{name}.txt'
                with open(texts_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(caption)
                        else:
                            new_image_list = image_list[int(f_tag*fps/15): int(to_tag*fps/15)]
                            if len(new_image_list) == 0:
                                continue
                            new_name = '%s_%f_%f'%(name, f_tag, to_tag)
                            self.data_dict[new_name] = {'images': torch.cat(new_image_list, dim=0),
                                                       'text':[caption]}
                            self.new_name_list.append(new_name)
                if flag:
                    self.data_dict[name] = {'images': torch.cat(image_list, dim=0),
                                            'text':text_data}
                    self.new_name_list.append(name)

    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, index):
        data = self.data_dict[self.new_name_list[index]]
        images, text_list = data['images'], data['text']
        text_data = random.choice(text_list)
        return images, text_data, len(images)

def collate_fn(batch):
    images = torch.cat([i[0] for i in batch], dim=0)
    texts = [i[1] for i in batch]
    index = torch.from_numpy(np.array([0] + [i[2] for i in batch]).cumsum(axis=-1))
    return {'images': images, 'texts': texts, 'index': index}

def DATALoader(split, batch_size, nodebug = False) : 

    num_workers = 8 if nodebug else 0
    train_loader = torch.utils.data.DataLoader(FinetuneDataset(split=split, nodebug=nodebug),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True)
    return train_loader