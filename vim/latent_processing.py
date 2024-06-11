import argparse
import os
import numpy as np
import torch
import torchvision
from datasets_prep import get_dataset
from tqdm import tqdm
from diffusers.models import AutoencoderKL
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel, PretrainedConfig
import pandas as pd
import functools
import random


    
    
class LatentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        transform = None
    ):
        self.path = path
        self.np_paths = [os.path.join(path, file) for file in os.listdir(path) if '0' in file and file.endswith('.npy')]
        self.transform = transform
    
    def __len__(self):
        return len(self.np_paths)
    
    def __getitem__(self, idx):
        npy_file = np.load(self.np_paths[idx], allow_pickle=True)
        latent = torch.from_numpy(x)
        if self.transform is not None:
            latent = self.transform
        return latent, 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Compute dataset stat')
    parser.add_argument('--datadir', default='./vim/dataset/celeba-lmdb')
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument('--dataset', default='celeba_256')
    parser.add_argument('--save_path', default='./vim/dataset/latent_celeba_256/')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='size of image')
    parser.add_argument('--image_size', type=int, default=256,
                        help='size of image')

    args = parser.parse_args()

    device = 'cuda:0'
    # Tuan Trung said bf16 more stable
    # fp32 to more precision according to lcm but bf16 for training to mixed precision
    weight_dtype = torch.float32
    # weight_dtype = torch.bfloat16
    os.makedirs(args.save_path, exist_ok=True)

    # 4. Load VAE checkpoint (or more stable VAE)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")

    dataset = get_dataset(args)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=4)
    
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    for i, (image, _) in enumerate(tqdm(dataloader)):       
        image = image.to(device, non_blocking=True)
        pixel_values = image.to(dtype=weight_dtype)
        if vae.dtype != weight_dtype:
            vae.to(dtype=weight_dtype)

        # encode pixel values with batch size of at most 32
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        latents = latents.to(weight_dtype)
        latents = latents.detach().cpu().numpy() # (bs, 4, 64, 64)
        
        for j in range(len(latents)):
            np.save(f'{args.save_path}/{str(i * args.batch_size + j).zfill(9)}.npy', latents[j])
        print('Generate batch {}'.format(i))
        


    # test
    debug_idex = list(torch.randint(0, len(dataset), (10,)).numpy())
    data = [np.load(f"{args.save_path}/{str(i).zfill(9)}.npy", allow_pickle=True) for i in debug_idex]
    sample = torch.stack([torch.from_numpy(x) for x in data])

    with torch.no_grad():
        rec_image = vae.decode(sample.cuda()/vae.config.scaling_factor).sample
    rec_image = torch.clamp((rec_image + 1.) / 2., 0, 1)
    torchvision.utils.save_image(rec_image, './rec_debug.jpg')