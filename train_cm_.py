# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import math
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
from models.random_util import get_generator
from models.script_util import (
    create_model_and_diffusion,
    create_ema_and_scales_fn,
)
from models.karras_diffusion import karras_sample
from models.network_dit import DiT_models
import json
from models.nn import mean_flat, append_dims, append_zero
from models.optimal_transport import OTPlanSampler

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    assert args.global_batch_size % world_size == 0, f"Batch size must be divisible by world size."
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup an experiment folder:
    experiment_index = args.exp
    experiment_dir = f"{args.results_dir}/{args.dataset}/{experiment_index}"  # Create an experiment folder 
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    sample_dir = f"{experiment_dir}/samples"
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    
    # create diffusion and model
    model, diffusion = create_model_and_diffusion(args)
    diffusion.c = 0.00054*math.sqrt(args.num_in_channels*args.image_size**2)
    logger.info("c in huber loss is {}".format(diffusion.c))
    # create ema for training model
    logger.info("creating the ema model")
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.to(device)
    # create target model
    logger.info("creating the target model")
    target_model = deepcopy(model).to(device)
    # target_model.requires_grad_(False)
    # target_model.train()
    
    # model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    # opt = torch.optim.RAdam(
    #     model.parameters(), lr=args.lr, #weight_decay=args.weight_decay
    # )

    if args.model_ckpt and os.path.exists(args.model_ckpt):
        checkpoint = torch.load(args.model_ckpt, map_location=torch.device(f'cuda:{device}'))
        epoch = init_epoch = checkpoint["epoch"]
        # model.module.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        # opt.load_state_dict(checkpoint["opt"])
        target_model.load_state_dict(checkpoint["target"])
        # for g in opt.param_groups:
        #     g['lr'] = args.lr
        train_steps = checkpoint["train_steps"]
        logger.info("=> loaded checkpoint (epoch {})".format(epoch))
        del checkpoint
    elif args.resume:
        checkpoint_file = os.path.join(checkpoint_dir, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(f'cuda:{device}'))
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        # model.module.load_state_dict(checkpoint["model"])
        model.load_state_dict(checkpoint["model"])
        # opt.load_state_dict(checkpoint["opt"])
        ema.load_state_dict(checkpoint["ema"])
        target_model.load_state_dict(checkpoint["target"])
        train_steps = checkpoint["train_steps"]
        logger.info("=> resume checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint
    else:
        init_epoch = 0
        train_steps = 0
    # requires_grad(ema, False)
    # ema.eval()
    model.eval(); model.requires_grad_(False)
    ema.eval(); ema.requires_grad_(False)
    target_model.eval(); target_model.requires_grad_(False)
    model.to(device)

    dataset = get_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.datadir})")
    args.total_training_steps = math.ceil(len(dataset)//args.global_batch_size)*args.epochs
    if rank == 0:
        config = vars(args)
        with open(f"{experiment_dir}/config.json", 'w') as out:
            json.dump(config, out)
    # create ema schedule
    logger.info("creating model and diffusion and ema scale function")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        ict=args.ict,
    ) # already adding ict increasing discretized N to code
    
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()
    use_label = True if "imagenet" in args.dataset else False
    
    if rank == 0:
        noise = torch.randn((64, 3, args.image_size, args.image_size), device=device)*args.sigma_max
    ema_rate, num_scales = ema_scale_fn(train_steps)

    logger.info(f"Training for {args.epochs} epochs which is {args.total_training_steps} iterations...")
    ckpt_file = '0000025.pt'
    ot_sampler = OTPlanSampler(method="exact", normalize_cost=True)
    os.makedirs('./loss_diffs', exist_ok=True)
    os.makedirs(os.path.join('./loss_diffs', f'cifar10_{num_scales}nfe_ckpt_{ckpt_file}'), exist_ok=True)
    for scale_value in tqdm(range(num_scales - 2, -1, -1), desc=f'{num_scales} nfe, ckpt {ckpt_file}'):
        os.makedirs(os.path.join('./loss_diffs', f'cifar10_{num_scales}nfe_ckpt_{ckpt_file}', f'{scale_value}'), exist_ok=True)
        for i, (x, y) in enumerate(tqdm(loader)):
            # adjust_learning_rate(opt, i / len(loader) + epoch, args)
            x = x.to(device)
            y = None if not use_label else y.to(device)
            noise = torch.randn_like(x)
            x, noise, y, _ = ot_sampler.sample_plan_with_labels(x0=x, x1=noise, y0=y, y1=None, replace=False)
            dims = x.ndim
            model_kwargs = dict(y=y)
            indices = torch.ones(size=(x.shape[0],), device=x.device) * scale_value
            t = diffusion.sigma_max ** (1 / diffusion.rho) + indices / (num_scales - 1) * (
                diffusion.sigma_min ** (1 / diffusion.rho) - diffusion.sigma_max ** (1 / diffusion.rho)
            )
            t = t**diffusion.rho
            x_t = x + noise * append_dims(t, dims)
            t2 = diffusion.sigma_max ** (1 / diffusion.rho) + (indices + 1) / (num_scales - 1) * (
                diffusion.sigma_min ** (1 / diffusion.rho) - diffusion.sigma_max ** (1 / diffusion.rho)
            )
            t2 = t2**diffusion.rho
            x_t2 = x + noise * append_dims(t2, dims)
            
            with torch.no_grad():
                dropout_state = torch.get_rng_state()
                model_output, distiller = diffusion.denoise(model, x_t, t, **model_kwargs)
                torch.set_rng_state(dropout_state)
                model_output_target, distiller_target = diffusion.denoise(model, x_t2, t2, **model_kwargs)
            diffs = distiller - distiller_target
            with open(os.path.join('./loss_diffs', f'cifar10_{num_scales}nfe_ckpt_{ckpt_file}', f'{scale_value}', f'{i}.npy'), 'wb') as f:
                np.save(f, diffs.cpu().numpy())


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    ###### misc ######
    parser.add_argument("--global-seed", type=int, default=0)
    ###### data specs ######
    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--num-in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    
    ###### model ######
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-heads-upsample", type=int, default=-1)
    parser.add_argument("--num-head-channels", type=int, default=-1)
    parser.add_argument("--attention-resolutions", type=str, default="32,16,8")
    parser.add_argument("--channel-mult", type=str, default="1,2,2,2")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-checkpoint", action="store_true", default=False)
    parser.add_argument("--use-scale-shift-norm", action="store_true", default=True)
    parser.add_argument("--resblock-updown", action="store_true", default=False)
    parser.add_argument("--use-fp16", action="store_true", default=False)
    parser.add_argument("--use-new-attention-order", action="store_true", default=False)
    parser.add_argument("--learn-sigma", action="store_true", default=False)
    parser.add_argument("--model-type", type=str, choices=["openai_unet", "song_unet", "dhariwal_unet"]+list(DiT_models.keys()), default="openai_unet")
    
    ###### diffusion ######
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--weight-schedule", type=str, choices=["karras", "snr", "snr+1", "uniform", "truncated-snr", "ict"], default="uniform")
    parser.add_argument("--noise-sampler", type=str, choices=["uniform", "ict"], default="ict")
    parser.add_argument("--loss-norm", type=str, choices=["l1", "l2", "lpips", "huber"], default="huber")
    
    ###### consistency ######
    parser.add_argument("--target-ema-mode", type=str, choices=["adaptive", "fixed"], default="fixed")
    parser.add_argument("--scale-mode", type=str, choices=["progressive", "fixed"], default="fixed")
    parser.add_argument("--start-ema", type=float, default=0.0)
    parser.add_argument("--start-scales", type=float, default=2)
    parser.add_argument("--end-scales", type=float, default=200)
    parser.add_argument("--ict", action="store_true", default=False)
    
    ###### training ######
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--no-lr-decay", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    
    ###### ploting & saving ######
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--save-content-every", type=int, default=5)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    
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
    parser.add_argument("--ts", type=str, default="0,22,39")

    args = parser.parse_args()
    main(args)