import torch
import options.option_finetune as option_finetune
from blip2.model.blip2_opt_arch import Blip2OPT
from omegaconf import OmegaConf
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import pdb
from glob import glob
from os.path import join as pjoin
import os

def get_transform(split):
    img_size = 224
    if split == "train":
        transform = T.Compose(
            [
                T.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.8, 1.0),
                    ratio=(1.0, 1.0),
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                lambda image: image.convert("RGB"),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize(img_size, interpolation=Image.BICUBIC),
                T.CenterCrop(img_size),
                lambda image: image.convert("RGB"),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
    return transform

def img_preprocess(path, transform):
    if path.endswith('.png'):
        image_files = [path]
    else:
        image_files = glob(pjoin(path, '*'))
    images = [transform(Image.open(file)).unsqueeze(0).cuda() for file in image_files]
    return {'image': images}

def get_test_paths(test_num):
    path_list = []
    with open('./dataset/KIT-ML/test.txt', 'r') as f:
        for i in range(test_num):
            label = f.readline()[:-1]
            path_list.append(f'./dataset/HumanML3D/render_frames/{label}_mesh_frames')
    return path_list
    
def main():
    args = option_finetune.get_args_parser()
    opt = OmegaConf.load(args.config_path)
    model = Blip2OPT(opt.model).cuda()
    transform = get_transform('test')
    
    save_file = open(args.save_path, 'w')
    # files = [f'/HOME/lyh/vq-mot/dataset/HumanML3D/render_frames/{i:06d}_mesh_frames' for i in range(100)]
    files = get_test_paths(100)
    for file_path in tqdm(files):
        if not os.path.exists(file_path):
            continue
        label = os.path.basename(file_path)[:6]
        samples = img_preprocess(file_path, transform)
        output_text = model(samples, train=False)
        msg = f'{label}: {output_text}\n'
        print(msg)
        save_file.write(msg)
        
if __name__ == "__main__":
    main()