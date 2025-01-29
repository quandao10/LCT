# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import math
import json
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from time import time
import argparse
import logging
import os
from datasets_prep import get_dataset
from tqdm import tqdm
from models.script_util import (
    create_model_and_diffusion,
    create_ema_and_scales_fn,
    create_model_umt,
)
from models.karras_diffusion import karras_sample
from diffusers.models import AutoencoderKL
from models.network_dit import DiT_models
from models.network_edm2 import EDM2_models
import robust_loss_pytorch
from sampler.random_util import get_generator
from models.optimal_transport import OTPlanSampler
from repa_utils import preprocess_raw_image

from repa_utils import load_encoders
encoders, encoder_types, architectures = load_encoders(args.enc_type, device)
z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]

def precompute_ssl_feat(args):
    device = torch.device("cuda")
    dataset = get_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)

    
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)

        ####################### REPA #######################
        ssl_feat = None
        if args.use_repa:
            with torch.no_grad():
                target = x.clone().detach()
                raw_image = target / vae.config.scaling_factor
                raw_image = vae.decode(raw_image.to(dtype=vae.dtype)).sample.float()
                raw_image = (raw_image * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                ssl_feat = []
                # with torch.autocast(device_type='cuda', dtype=__dtype):
                with torch.autocast(device_type='cuda'):
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type: z = z = z[:, 1:] 
                        if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                        ssl_feat.append(z.detach())
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=str, default="/home/khanhdn10/repo/lct/dataset/")
    parser.add_argument("--dataset", type=str, default="latent_celeb256")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--enc_type", type=str, default="dinov2-vit-b")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    precompute_ssl_feat(args)