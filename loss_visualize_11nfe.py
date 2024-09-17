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
from datasets_prep import get_dataset, CustomDataset
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

from models.nn import mean_flat, append_dims, append_zero
from torchvision.utils import make_grid, save_image
from PIL import Image

# Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
parser = argparse.ArgumentParser()
###### misc ######
parser.add_argument("--global-seed", type=int, default=0)
###### data specs ######
parser.add_argument('--dataset', default='latent_celeb256', help='name of dataset')
parser.add_argument("--datadir", type=str, default='./dataset')
parser.add_argument("--image-size", type=int, default=32)
parser.add_argument("--num-in-channels", type=int, default=4)
parser.add_argument("--num-classes", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--normalize-matrix", type=str, default='./celeb256_stat.npy')

###### model ######
parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
parser.add_argument("--num-channels", type=int, default=128)
parser.add_argument("--num-res-blocks", type=int, default=4)
parser.add_argument("--num-heads", type=int, default=4)
parser.add_argument("--num-heads-upsample", type=int, default=-1)
parser.add_argument("--num-head-channels", type=int, default=64)
parser.add_argument("--attention-resolutions", type=str, default="16,8")
parser.add_argument("--channel-mult", type=str, default="1,2,3,4")
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--use-checkpoint", action="store_true", default=False)
parser.add_argument("--use-scale-shift-norm", action="store_true", default=True)
parser.add_argument("--resblock-updown", action="store_true", default=True)
parser.add_argument("--use-fp16", action="store_true", default=False)
parser.add_argument("--use-new-attention-order", action="store_true", default=False)
parser.add_argument("--learn-sigma", action="store_true", default=False)
parser.add_argument("--model-type", type=str, choices=["openai_unet", "song_unet", "dhariwal_unet"]+list(DiT_models.keys())+list(EDM2_models.keys()), default="dhariwal_unet")

###### diffusion ######
parser.add_argument("--sigma-min", type=float, default=0.002)
parser.add_argument("--sigma-max", type=float, default=80.0)
parser.add_argument("--weight-schedule", type=str, choices=["karras", "snr", "snr+1", "uniform", "truncated-snr", "ict"], default="ict")
parser.add_argument("--noise-sampler", type=str, choices=["uniform", "ict"], default="ict")
parser.add_argument("--loss-norm", type=str, choices=["l1", "l2", "lpips", "huber", "adaptive", "cauchy", "gm"], default="cauchy")
parser.add_argument("--ot-hard", action="store_true", default=True)

###### consistency ######
parser.add_argument("--target-ema-mode", type=str, choices=["adaptive", "fixed"], default="adaptive")
parser.add_argument("--scale-mode", type=str, choices=["progressive", "fixed"], default="progressive")
parser.add_argument("--start-ema", type=float, default=0.95)
parser.add_argument("--start-scales", type=float, default=10)
parser.add_argument("--end-scales", type=float, default=640)
parser.add_argument("--ict", action="store_true", default=True)
parser.add_argument("--l2-reweight", action="store_true", default=False)
parser.add_argument("--use-diffloss", action="store_true", default=True)
parser.add_argument("--ema-half-nfe", action="store_true", default=False)
parser.add_argument("--umt", help="Uncertainty-based multi-task learning", action="store_true", default=False)

###### training ######
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--no-lr-decay", action='store_true', default=False)
parser.add_argument("--epochs", type=int, default=1400)
parser.add_argument("--global-batch-size", type=int, default=8)
parser.add_argument("--max-grad-norm", type=float, default=100.0)


###### ploting & saving ######
parser.add_argument("--log-every", type=int, default=100)
parser.add_argument("--ckpt-every", type=int, default=25)
parser.add_argument("--save-content-every", type=int, default=5)
parser.add_argument("--plot-every", type=int, default=5)
parser.add_argument("--exp", type=str, default="large_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard")
parser.add_argument("--results-dir", type=str, default="./results")

###### resume ######
parser.add_argument("--model-ckpt", type=str, default='')
parser.add_argument("--resume", action="store_true", default=False)

###### sampling ######
parser.add_argument("--cfg-scale", type=float, default=1.)
parser.add_argument("--clip-denoised", action="store_true", default=True)
parser.add_argument("--sampler", type=str, default='onestep')
parser.add_argument("--s-churn", type=float, default=0.0)
parser.add_argument("--s-tmin", type=float, default=0.0)
parser.add_argument("--s-tmax", type=float, default=float("inf"))
parser.add_argument("--s-noise", type=float, default=1.0)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--num-sampling", type=int, default=8)
parser.add_argument("--ts", type=str, default="0,22,39")

args, unknown = parser.parse_known_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')
seed = args.global_seed
torch.manual_seed(seed)

# Setup an experiment folder:
experiment_index = args.exp
experiment_dir = f"{args.results_dir}/{args.dataset}/{experiment_index}"  # Create an experiment folder 
checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
sample_dir = f"{experiment_dir}/samples"

vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
model, diffusion = create_model_and_diffusion(args)
diffusion.c = torch.tensor(0.00054*math.sqrt(args.num_in_channels*args.image_size**2))
model.to(device)
ema = deepcopy(model).to(device)
target_model = deepcopy(model).to(device)
# target_model.requires_grad_(False)
# target_model.train()

vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
ot_sampler = OTPlanSampler(method="exact", normalize_cost=True)
data = np.load(args.normalize_matrix, allow_pickle=True).item()
mean = data["mean"].to(device)
std = data["std"].to(device)

dataset = CustomDataset("celebhq256", "./dataset/latent_celeba_256")
loader = DataLoader(
    dataset,
    # batch_size=int(args.global_batch_size // 1),
    batch_size=128,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=False
)

num_scales = 11

ckpt_files = [
    '0000025.pt',
    '0000050.pt',
    '0000075.pt',
    '0000100.pt',
    '0000125.pt',
    '0000150.pt',
    '0000175.pt',
]

for ckpt_file in ckpt_files:
    args.model_ckpt = os.path.join(checkpoint_dir, ckpt_file)
    checkpoint = torch.load(args.model_ckpt, map_location=device)
    epoch = init_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    target_model.load_state_dict(checkpoint["target"])
    train_steps = checkpoint["train_steps"]

    model.eval(); model.requires_grad_(False)
    ema.eval(); ema.requires_grad_(False)
    target_model.eval(); target_model.requires_grad_(False)

    use_label = True if "imagenet" in args.dataset else False
    results = dict()
    for scale_value in tqdm(range(num_scales - 2, -1, -1), desc=f'{num_scales}nfe, ckpt {ckpt_file}'):
        count = 0
        var = 1.0
        mean = 0.0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            x = x/0.18215
            x = (x - mean)/std * 0.5
            y = None if not use_label else y.to(device)
            n = torch.randn_like(x)
            x, n, y, _ = ot_sampler.sample_plan_with_labels(x0=x, x1=n, y0=y, y1=None, replace=False)
            dims = x.ndim
            model_kwargs = dict(y=y)
            indices = torch.zeros(size=(x.shape[0],), device=x.device) * scale_value
            t = diffusion.sigma_max ** (1 / diffusion.rho) + indices / (num_scales - 1) * (
                diffusion.sigma_min ** (1 / diffusion.rho) - diffusion.sigma_max ** (1 / diffusion.rho)
            )
            t = t**diffusion.rho
            x_t = x + n * append_dims(t, dims)
            t2 = diffusion.sigma_max ** (1 / diffusion.rho) + (indices + 1) / (num_scales - 1) * (
                diffusion.sigma_min ** (1 / diffusion.rho) - diffusion.sigma_max ** (1 / diffusion.rho)
            )
            t2 = t2**diffusion.rho
            x_t2 = x + n * append_dims(t2, dims)
            
            with torch.no_grad():
                dropout_state = torch.get_rng_state()
                distiller = diffusion.denoise(model, x_t, t, **model_kwargs)[1]
                torch.set_rng_state(dropout_state)
                distiller_target = diffusion.denoise(model, x_t2, t2, **model_kwargs)[1]
            diffs = distiller - distiller_target
            
            batch_mean = diffs.mean().item()
            batch_var = diffs.var().item()
            
            # update count
            new_coming = x.numel()
            count += new_coming
            
            # update mean
            delta = batch_mean - mean
            mean += delta * new_coming / count
            
            # update var
            m_a = var * (count - new_coming)
            m_b = batch_var * new_coming
            M2 = m_a + m_b + delta**2 * new_coming
            var = M2 / count
        std = var**0.5
        results[scale_value] = {
            'mean': mean,
            'var': var,
            'std': std,
        }

    f = open(f'./loss_visualize_{num_scales}nfe_ckpt_{ckpt_file}.jsonl', 'w')
    json.dump(results, f, indent=4)
    f.close()
