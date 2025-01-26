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
from models.network_udit import UDiT_models
from models.network_edm2 import EDM2_models
import robust_loss_pytorch
from sampler.random_util import get_generator
from models.optimal_transport import OTPlanSampler
from lion_pytorch import Lion
from efficientvit.efficientvit.ae_model_zoo import DCAE_HF
from soap import SOAP
#################################################################################
#                             Training Helper Functions                         #
#################################################################################


# EMA_RATES = {
#     "ema_0.999": 0.999,
#     "ema": 0.9999,
#     "ema_0.99993": 0.99993,
#     "ema_0.99994": 0.99994,
#     "ema_0.99995": 0.99995,
#     "ema_0.99997": 0.99997,
#     "ema_0.9999432189950708": 0.9999432189950708,
# }

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

def norm_dim(x):
    _, C, H, W = x.shape
    return torch.sqrt(torch.sum(x**2, dim=(1,2,3))/(C*H*W))


#################################################################################
#                                  Training Loop                                #
#################################################################################

nfe_to_c = {
    11: 0.08,
    21: 0.04,
    41: 0.015,
    81: 0.008,
    161: 0.004,
    321: 0.004,
    641: 0.004
}

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
    # create vae model
    logger.info("creating the vae model")
    # vae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f32c32-in-1.0").to(device).eval()
    def build_vae(args):
        if "mit-han-lab/dc-ae-f32c32-in-1.0" in args.vae_type:
            vae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f32c32-in-1.0")
        elif "stabilityai/sd-vae-ft-ema" in args.vae_type:
            vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
        else:
            raise ValueError(f"VAE type {args.vae_type} not supported")
        
        print(f"\033[33mLoaded vae: {args.vae_type}\033[0m")
        return vae
    
    vae = build_vae(args).to(device).eval()
    vae.requires_grad_(False)
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    # create diffusion and model
    # repa
    if args.use_repa:
        from repa_utils import load_encoders
        encoders, encoder_types, architectures = load_encoders(args.enc_type, device)
        z_dims = [encoder.embed_dim for encoder in encoders] if args.enc_type != 'None' else [0]
        args.z_dims = z_dims
        
    model, diffusion = create_model_and_diffusion(args)
    if args.custom_constant_c > 0.0:
        diffusion.c = torch.tensor(args.custom_constant_c)
    else:
        diffusion.c = torch.tensor(0.00054*math.sqrt(args.num_in_channels*args.image_size**2))
    # diffusion.c = torch.tensor(0.00345)
    logger.info("c in huber loss is {}".format(diffusion.c.item()))
    # create ema for training model
    logger.info("creating the ema model")
    ema = deepcopy(model)  # Create an EMA of the model for use after training
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.to(device)
    # create target model
    logger.info("creating the target model")
    target_model = deepcopy(model).to(device)
    target_model.requires_grad_(False)
    target_model.train()
    
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)

    # Uncertainty-based multi-task learning
    if args.umt:
        model_umt = create_model_umt(args)
        model_umt = DDP(model_umt.to(device), device_ids=[rank], find_unused_parameters=False)
    else:
        model_umt = None
    
    if args.loss_norm=="adaptive":
        adaptive_loss = robust_loss_pytorch.adaptive.AdaptiveLossFunction(num_dims=args.num_in_channels*args.image_size**2, 
                                                                          alpha_hi=1.0,
                                                                          alpha_lo=0.0,
                                                                          float_dtype=np.float32, 
                                                                          scale_init=diffusion.c,
                                                                          device=device)
        if args.umt:
            opt = torch.optim.RAdam(list(model.parameters())+list(adaptive_loss.parameters())+list(model_umt.parameters()), lr=args.lr, weight_decay=1e-4)
        else:
            opt = torch.optim.RAdam(list(model.parameters())+list(adaptive_loss.parameters()), lr=args.lr, weight_decay=1e-4)
    else:
        adaptive_loss = None
        if args.umt:
            opt = torch.optim.RAdam(list(model.parameters())+list(model_umt.parameters()), lr=args.lr, weight_decay=1e-4)
        else:
            # opt = torch.optim.RAdam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=args.eps)
            opt = SOAP(model.parameters(), lr=args.lr, betas=(.95, .95), weight_decay=.01, precondition_frequency=10, eps=args.eps)
    # define scheduler
    if args.model_ckpt and os.path.exists(args.model_ckpt):
        checkpoint = torch.load(args.model_ckpt, map_location=torch.device(f'cuda:{device}'))
        epoch = init_epoch = checkpoint["epoch"]
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        opt.load_state_dict(checkpoint["opt"])
        target_model.load_state_dict(checkpoint["target"])
        train_steps = checkpoint["train_steps"]
        if args.umt:
            model_umt.module.load_state_dict(checkpoint["model_umt"])
        logger.info("=> loaded checkpoint (epoch {})".format(epoch))
        del checkpoint
    elif args.resume:
        checkpoint_file = os.path.join(checkpoint_dir, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(f'cuda:{device}'))
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        model.module.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        ema.load_state_dict(checkpoint["ema"])
        target_model.load_state_dict(checkpoint["target"])
        train_steps = checkpoint["train_steps"]
        if args.umt:
            model_umt.module.load_state_dict(checkpoint["model_umt"])
        logger.info("=> resume checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint
    else:
        init_epoch = 0
        train_steps = 0
    requires_grad(ema, False)
    ema.eval()

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
    )

    # OT sampler
    if args.ot_hard:
        ot_sampler = OTPlanSampler(method="exact", normalize_cost=True)
    else:
        ot_sampler = None
    
    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_cm_loss = 0
    running_diff_loss = 0
    start_time = time()
    nan_count = 0
    use_label = True if "imagenet" in args.dataset else False
    # load normalize matrix
    use_normalize = args.normalize_matrix is not None
    if use_normalize:
        data = np.load(args.normalize_matrix, allow_pickle=True).item()
        mean = data["mean"].to(device)
        std = data["std"].to(device)
        
    if rank == 0:
        # Create sampling noise & label
        ys = None if not use_label else torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], device=device)
        bsz = args.num_sampling if not use_label else ys.shape[0]
        noise = torch.randn((bsz, args.num_in_channels, args.image_size, args.image_size), device=device)*args.sigma_max
        use_cfg = args.cfg_scale > 1.0
        # Setup classifier-free guidance:
        if use_cfg and use_label:
            noise = torch.cat([noise, noise], 0)
            y_null = torch.tensor([args.num_classes] * bsz, device=device)
            ys = torch.cat([ys, y_null], 0)
            sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
            model_fn = ema.forward_with_cfg
        else:
            sample_model_kwargs = dict(y=ys)
            model_fn = ema.forward

    logger.info(f"Training for {args.epochs} epochs which is {args.total_training_steps} iterations...")
    for epoch in range(init_epoch, args.epochs+1):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for i, (x, y) in enumerate(tqdm(loader)):
            # adjust_learning_rate(opt, i / len(loader) + epoch, args)
            x = x.to(device)
            if use_normalize:
                x = x#/0.18215
                x = (x - mean)/std * 0.5
            y = None if not use_label else y.to(device)
            n = torch.randn_like(x)
            if args.ot_hard:
                x, n, y, _ = ot_sampler.sample_plan_with_labels(x0=x, x1=n, y0=y, y1=None, replace=False)
            ema_rate, num_scales = ema_scale_fn(train_steps)

            # Change diffusion.c w.r.t predicted function by NFE (loss std)
            if args.c_by_loss_std and (num_scales > (args.start_scales + 1)): # From the second NFE scale
                diffusion.c = torch.tensor(math.exp(-1.15 * math.log(float(num_scales - 1)) - 0.85)) * torch.sqrt(torch.tensor(2))
            model_kwargs = dict(y=y)
            before_forward = torch.cuda.memory_allocated(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                losses = diffusion.consistency_losses(model,
                                                    x,
                                                    num_scales,
                                                    target_model=target_model,
                                                    model_kwargs=model_kwargs,
                                                    noise=n,
                                                    adaptive=adaptive_loss,
                                                    model_umt=model_umt)
           
            if args.l2_reweight:
                distances = norm_dim(x-n)
                weight = (distances-distances.min())/(distances.max()-distances.min()) + 0.1
                loss = (losses["loss"]*weight).mean()
            else:
                cm_loss = losses["loss"].mean() 
                diff_loss = losses["diff_loss"].mean()
                if args.use_diffloss:
                    loss = cm_loss + args.diff_lamb*diff_loss
                else:
                    loss = cm_loss
                    diff_loss = torch.tensor(0)
            after_forward = torch.cuda.memory_allocated(device)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            for param in model.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad) # this is interesting
            opt.step()
            running_loss += loss.item()
            running_cm_loss += cm_loss.item()
            running_diff_loss += diff_loss.item()
            after_backward = torch.cuda.memory_allocated(device)
            update_ema(ema, model.module)
            ##### ema rate for teacher should be 0 (iCT) because our bs is small, we might not need set ema = 0 (more unstable)
            if args.ict:
                update_ema(target_model, model.module, 0)
            else:
                update_ema(target_model, model.module, ema_rate)

            # Log loss values:
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_cm_loss = torch.tensor(running_cm_loss / log_steps, device=device)
                avg_diff_loss = torch.tensor(running_diff_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(
                    f"(step={train_steps:07d}, nfe={num_scales}, c={diffusion.c}) Train Loss: {avg_loss:.4f} CM Loss: {avg_cm_loss:.4f} Diff Loss: {avg_diff_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"GPU Mem before forward: {before_forward/10**9:.2f}Gb, "
                    f"GPU Mem after forward: {after_forward/10**9:.2f}Gb, "
                    f"GPU Mem after backward: {after_backward/10**9:.2f}Gb"
                )
                # Reset monitoring variables:
                running_loss = 0
                running_cm_loss = 0
                running_diff_loss = 0
                log_steps = 0
                start_time = time()
            if rank == 0 and False and (train_steps % args.plot_every == 0): #epoch % args.plot_every == 0:
                logger.info("Generating EMA samples...")
                generator = get_generator("dummy", 4, seed)
                if args.sampler == "multistep":
                    assert len(args.ts) > 0
                    ts = tuple(int(x) for x in args.ts.split(","))
                else:
                    ts = None
                with torch.no_grad():
                    sample = karras_sample(
                        diffusion,
                        generator,
                        model.module.forward_with_cfg if use_cfg else model,
                        (args.num_sampling, args.num_in_channels, args.image_size, args.image_size),
                        steps=args.steps,
                        model_kwargs=sample_model_kwargs, # put sample model arg here
                        device=device,
                        clip_denoised=args.clip_denoised,
                        sampler=args.sampler,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        s_churn=args.s_churn,
                        s_tmin=args.s_tmin,
                        s_tmax=args.s_tmax,
                        s_noise=args.s_noise,
                        noise=noise,
                        ts=ts,
                    )
                    if use_cfg:
                        sample, _ = sample.chunk(2, dim=0)
                    if use_normalize:
                        sample = [vae.decode(x.unsqueeze(0)*std/0.5 + mean).sample for x in sample]
                    else:
                        sample = [vae.decode(x.unsqueeze(0) / 0.18215).sample for x in sample]
                sample = torch.concat(sample, dim=0)
                with torch.no_grad():
                    ema_sample = karras_sample(
                        diffusion,
                        generator,
                        model_fn,
                        (args.num_sampling, args.num_in_channels, args.image_size, args.image_size),
                        steps=args.steps,
                        model_kwargs=sample_model_kwargs,
                        device=device,
                        clip_denoised=args.clip_denoised,
                        sampler=args.sampler,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        s_churn=args.s_churn,
                        s_tmin=args.s_tmin,
                        s_tmax=args.s_tmax,
                        s_noise=args.s_noise,
                        noise=noise,
                        ts=ts,
                    )
                    if use_cfg:
                        ema_sample, _ = ema_sample.chunk(2, dim=0)
                    if use_normalize:
                        ema_sample = [vae.decode(x.unsqueeze(0)*std/0.5 + mean).sample for x in ema_sample]
                    else:
                        ema_sample = [vae.decode(x.unsqueeze(0) / 0.18215).sample for x in ema_sample]
                ema_sample = torch.concat(ema_sample, dim=0)
                sample_to_save = torch.concat([sample, ema_sample], dim=0)
                save_image(sample_to_save, f"{sample_dir}/image_{train_steps:07d}.jpg", nrow=8, normalize=True, value_range=(-1, 1))
                del sample

        # if not args.no_lr_decay:
        #     scheduler.step()

        if rank == 0:
            # latest checkpoint
            if epoch % args.save_content_every == 0:
                logger.info("Saving content.")
                content = {
                    "epoch": epoch + 1,
                    "train_steps": train_steps,
                    "args": args,
                    "model": model.module.state_dict(),
                    "opt": opt.state_dict(),
                    "ema": ema.state_dict(),
                    "target": target_model.state_dict(),
                    "model_umt": model_umt.module.state_dict() if args.umt else None,
                }
                torch.save(content, os.path.join(checkpoint_dir, "content.pth"))

            # Save DiT checkpoint:
            if epoch % args.ckpt_every == 0 and epoch > 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "model": model.module.state_dict(),
                    "train_steps": train_steps,
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                    "target": target_model.state_dict(),
                    "model_umt": model_umt.module.state_dict() if args.umt else None,
                }
                checkpoint_path = f"{checkpoint_dir}/{epoch:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            # dist.barrier()
            
            if rank == 0 and epoch % args.plot_every == 0 and True: #epoch % args.plot_every == 0:
                logger.info("Generating EMA samples...")
                generator = get_generator("dummy", 4, seed)
                if args.sampler == "multistep":
                    assert len(args.ts) > 0
                    ts = tuple(int(x) for x in args.ts.split(","))
                else:
                    ts = None
                with torch.no_grad():
                    sample = karras_sample(
                        diffusion,
                        generator,
                        model,
                        (args.num_sampling, args.num_in_channels, args.image_size, args.image_size),
                        steps=args.steps,
                        model_kwargs=sample_model_kwargs, # put sample model arg here
                        device=device,
                        clip_denoised=args.clip_denoised,
                        sampler=args.sampler,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        s_churn=args.s_churn,
                        s_tmin=args.s_tmin,
                        s_tmax=args.s_tmax,
                        s_noise=args.s_noise,
                        noise=noise,
                        ts=ts,
                    )
                    if use_normalize:
                        sample = [vae.decode(x.unsqueeze(0)*std/0.5 + mean) for x in sample]
                    else:
                        sample = [vae.decode(x.unsqueeze(0) / 0.18215).sample for x in sample]
                sample = torch.concat(sample, dim=0)
                with torch.no_grad():
                    ema_sample = karras_sample(
                        diffusion,
                        generator,
                        model_fn,
                        (args.num_sampling, args.num_in_channels, args.image_size, args.image_size),
                        steps=args.steps,
                        model_kwargs=sample_model_kwargs,
                        device=device,
                        clip_denoised=args.clip_denoised,
                        sampler=args.sampler,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        s_churn=args.s_churn,
                        s_tmin=args.s_tmin,
                        s_tmax=args.s_tmax,
                        s_noise=args.s_noise,
                        noise=noise,
                        ts=ts,
                    )
                    if use_normalize:
                        ema_sample = [vae.decode(x.unsqueeze(0)*std/0.5 + mean) for x in ema_sample]
                    else:
                        ema_sample = [vae.decode(x.unsqueeze(0) / 0.18215).sample for x in ema_sample]
                ema_sample = torch.concat(ema_sample, dim=0)
                sample_to_save = torch.concat([sample, ema_sample], dim=0)
                save_image(sample_to_save, f"{sample_dir}/image_{epoch:07d}.jpg", nrow=8, normalize=True, value_range=(-1, 1))
                del sample

        
        # dist.barrier()
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Done!")
    cleanup()


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
    parser.add_argument("--normalize-matrix", type=str, default=None)
    ###### model ######
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--vae-type", type=str, default="")
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
    parser.add_argument("--model-type", type=str, choices=["openai_unet", "song_unet", "dhariwal_unet"]+list(DiT_models.keys())+list(EDM2_models.keys())+list(UDiT_models.keys()), default="openai_unet")
    parser.add_argument("--no-scale", action="store_true", default=False)
    parser.add_argument("--wo-norm", action="store_true", default=False)
    parser.add_argument("--attn-type", type=str, default="normal")
    parser.add_argument("--linear-act", type=str, default=None)
    parser.add_argument("--num-register", type=int, default=0)
    parser.add_argument("--final-conv", action="store_true", default=False)
    
    
    ###### diffusion ######
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--weight-schedule", type=str, choices=["karras", "snr", "snr+1", "uniform", "truncated-snr", "ict", "log_ict", "norm_ict"], default="uniform")
    parser.add_argument("--noise-sampler", type=str, choices=["uniform", "ict"], default="ict")
    parser.add_argument("--loss-norm", type=str, choices=["l1", "l2", "lpips", "huber", "adaptive", "cauchy", "gm", "huber_new", "cauchy_new", "gm_new"], default="huber")
    parser.add_argument("--ot-hard", action="store_true", default=False)
    parser.add_argument("--c-by-loss-std", action="store_true", default=False)
    parser.add_argument("--custom-constant-c", type=float, default=0.0)
    parser.add_argument("--diff-lamb", type=float, default=5.0)
    
    
    ###### consistency ######
    parser.add_argument("--target-ema-mode", type=str, choices=["adaptive", "fixed"], default="fixed")
    parser.add_argument("--scale-mode", type=str, choices=["progressive", "fixed"], default="fixed")
    parser.add_argument("--start-ema", type=float, default=0.0)
    parser.add_argument("--start-scales", type=float, default=40)
    parser.add_argument("--end-scales", type=float, default=40)
    parser.add_argument("--ict", action="store_true", default=False)
    parser.add_argument("--l2-reweight", action="store_true", default=False)
    parser.add_argument("--use-diffloss", action="store_true", default=False)
    parser.add_argument("--ema-half-nfe", action="store_true", default=False)
    parser.add_argument("--umt", help="Uncertainty-based multi-task learning", action="store_true", default=False)
    parser.add_argument("--use-repa", action="store_true", default=False)
    parser.add_argument("--enc-type", type=str, default="vit_b_16")
    parser.add_argument("--projector-dim", type=int, default=2048)
    parser.add_argument("--encoder-depth", type=int, default=4)
    
    ###### training ######
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--no-lr-decay", action='store_true', default=False)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--max-grad-norm", type=float, default=2.0)
    
    
    ###### ploting & saving ######
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--save-content-every", type=int, default=5)
    parser.add_argument("--plot-every", type=int, default=1000)
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    
    ###### resume ######
    parser.add_argument("--model-ckpt", type=str, default='')
    parser.add_argument("--resume", action="store_true", default=False)
    
    ###### sampling ######
    parser.add_argument("--cfg-scale", type=float, default=1.)
    parser.add_argument("--clip-denoised", action="store_true", default=False)
    parser.add_argument("--sampler", type=str, default='onestep')
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--s-tmin", type=float, default=0.0)
    parser.add_argument("--s-tmax", type=float, default=float("inf"))
    parser.add_argument("--s-noise", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--num-sampling", type=int, default=4)
    parser.add_argument("--ts", type=str, default="0,22,39")

    args = parser.parse_args()
    main(args)