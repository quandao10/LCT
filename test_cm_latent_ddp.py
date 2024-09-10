# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import math
import os
import torchvision
import torch
import torch.distributed as dist
from diffusers.models import AutoencoderKL
from PIL import Image
from eval_toolbox.pytorch_fid.fid_score import calculate_fid_given_paths
from sampler.random_util import get_generator
from tqdm import tqdm
from models.network_dit import DiT_models
from models.script_util import (
    create_model_and_diffusion,
)
from models.karras_diffusion import karras_sample
import numpy as np
from datasets_prep import get_dataset
from torch.utils.data import DataLoader
from models.nn import mean_flat, append_dims, append_zero
from tqdm import tqdm
from models.karras_diffusion import get_sigmas_karras

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    seed = args.seed + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.dataset == "cifar10":
        real_img_dir = "pytorch_fid/cifar10_train_stat.npy"
    elif args.dataset == "celeba_256":
        real_img_dir = "pytorch_fid/celebahq_stat.npy"
    elif args.dataset == "lsun_church":
        real_img_dir = "pytorch_fid/lsun_church_stat.npy"
    elif args.dataset == "ffhq_256":
        real_img_dir = "pytorch_fid/ffhq_stat.npy"
    elif args.dataset == "lsun_bedroom":
        real_img_dir = "pytorch_fid/lsun_bedroom_stat.npy"
    elif args.dataset in ["latent_imagenet_256", "imagenet_256"]:
        real_img_dir = "pytorch_fid/imagenet_stat.npy"
    else:
        real_img_dir = args.real_img_dir

    to_range_0_1 = lambda x: (x + 1.0) / 2.0

    model, diffusion = create_model_and_diffusion(args)
    model.to(device=device)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    ckpt = torch.load(args.ckpt)
    print("Finish loading model")
    # loading weights from ddp in single gpu
    if not args.ema:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt["ema"], strict=True)
    model.eval()
    
    del ckpt
    args.exp = args.ckpt.split("/")[-3]
    args.epoch_id = args.ckpt.split("/")[-1][:-3]
    save_dir = "./generated_samples/{}/exp{}_ep{}".format(args.dataset, args.exp, args.epoch_id)
    
    if args.cfg_scale > 1.0:
        save_dir += "_cfg{}".format(args.cfg_scale)
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # seed generator
    # seed should be aligned with rank
    generator = get_generator(args.generator, args.num_sampling, seed)
    use_label = True if "imagenet" in args.dataset else False
    use_normalize = args.normalize_matrix is not None
    if use_normalize:
        data = np.load(args.normalize_matrix, allow_pickle=True).item()
        mean = data["mean"].to(device)
        std = data["std"].to(device)

    def run_sampling(num_samples, generator):
        noise = generator.randn(num_samples, 4, args.image_size, args.image_size).to(device)*args.sigma_max
        if not use_label:
            model_kwargs = dict(y=None)
        else:
            y = generator.randint(0, args.num_classes, (num_samples,), device=device)
            # Setup classifier-free guidance:
            if args.cfg_scale > 1.0:
                noise = torch.cat([noise, noise], 0)
                y_null = (torch.tensor([args.num_classes] * num_samples, device=device))
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            else:
                model_kwargs = dict(y=y)

        if args.sampler == "multistep":
            assert len(args.ts) > 0
            ts = tuple(int(x) for x in args.ts.split(","))
        else:
            ts = None
        with torch.no_grad():
            fake_sample = karras_sample(
                        diffusion,
                        generator,
                        model,
                        (args.batch_size, args.num_in_channels, args.image_size, args.image_size),
                        steps=args.steps,
                        model_kwargs=model_kwargs,
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
        if args.cfg_scale > 1.0:
            fake_sample, _ = fake_sample.chunk(2, dim=0)  # Remove null class samples        
        if use_normalize:
            fake_image = [vae.decode(x.unsqueeze(0)*std/0.5 + mean).sample for x in fake_sample] # careful here
        else:
            fake_image = [vae.decode(x.unsqueeze(0) / 0.18215).sample for x in fake_sample]
        return fake_image
    
    if args.compute_fid:
        print("Compute fid")
        dist.barrier()
        # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
        n = args.batch_size
        global_batch_size = n * dist.get_world_size()
        total_samples = int(math.ceil(args.num_sampling / global_batch_size) * global_batch_size)
        if rank == 0:
            print(f"Total number of images that will be sampled: {total_samples}")
        assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // dist.get_world_size())
        iters_needed = int(samples_needed_this_gpu // n)
        pbar = range(iters_needed)
        pbar = tqdm(pbar) if rank == 0 else pbar
        total = 0

        for i in pbar:
            with torch.no_grad():
                fake_image = run_sampling(args.batch_size, generator)
                fake_image = torch.cat(fake_image)
                fake_image = (
                    (torch.clamp(to_range_0_1(fake_image), 0, 1) * 255.0)
                    .permute(0, 2, 3, 1)
                    .to("cpu", dtype=torch.uint8)
                    .numpy()
                )
                for j, x in enumerate(fake_image):
                    index = j * dist.get_world_size() + rank + total
                    Image.fromarray(x).save(f"{save_dir}/{index}.jpg")
                if rank == 0:
                    print("generating batch ", i)
                total += global_batch_size
        # make sure all processes have finished
        dist.barrier()
        if rank == 0:
            paths = [save_dir, real_img_dir]
            kwargs = {"batch_size": 200, "device": device, "dims": 2048}
            fid = calculate_fid_given_paths(paths=paths, **kwargs)
            print("FID = {}".format(fid))
            with open(args.output_log, "a") as f:
                f.write("Epoch = {}, FID = {}, cfg_scale = {}\n".format(args.epoch_id, fid, args.cfg_scale))
        dist.barrier()
        dist.destroy_process_group()
    else:
        fake_image = run_sampling(args.batch_size, generator)
        fake_image = torch.cat(fake_image)
        ema = "" if not args.ema else "_ema"
        torchvision.utils.save_image(fake_image, f'{args.exp}_{args.epoch_id}{ema}.jpg', nrow=4, normalize=True, value_range=(-1, 1))
        dist.barrier()
        dist.destroy_process_group()
        
    # if args.test_interval:
    #     test_dir = "./test_x0"
    #     os.makedirs(test_dir, exist_ok=True)
    #     dataset = get_dataset(args)
    #     loader = DataLoader(
    #         dataset,
    #         batch_size=4,
    #         shuffle=False,
    #         num_workers=4,
    #         pin_memory=True,
    #         drop_last=True
    #     )
    #     args.rho = 7
    #     num_scales = 160
    #     image, _ = next(iter(loader))
    #     image = image.to(device)
    #     if use_normalize:
    #         image = image/0.18215
    #         image = (image - mean)/std
    #     dims = image.ndim
    #     noise = torch.randn_like(image)
    #     image_last = image + noise * append_dims(0.002*torch.ones((4,), device=device), dims)
    #     print("#####################")
    #     print(get_sigmas_karras(num_scales, args.sigma_min, args.sigma_max, args.rho))
    #     print("#####################")
    #     if use_normalize:
    #         ori_image = [vae.decode(x.unsqueeze(0)*std + mean).sample for x in image]
    #         image_last = [vae.decode(x.unsqueeze(0)*std + mean).sample for x in image_last]
    #     else:
    #         ori_image = [vae.decode(x.unsqueeze(0) / 0.18215).sample for x in image]
    #         image_last = [vae.decode(x.unsqueeze(0) / 0.18215).sample for x in image_last]
    #     ori_image = torch.cat(ori_image)
    #     image_last = torch.cat(image_last)
    #     torchvision.utils.save_image(ori_image, f"{test_dir}/x0_t=original.png", normalize=True, value_range=(-1, 1))
    #     torchvision.utils.save_image(image_last, f"{test_dir}/x0_t=last.png", normalize=True, value_range=(-1, 1))
    #     # indices = torch.randint(0, num_scales - 1, (image.shape[0],), device=image.device)
    #     for ind in tqdm(range(159, -1, -10)):
    #         indices = ind*torch.ones(size=(4, ), device=device)
    #         t = args.sigma_max ** (1 / args.rho) + indices / (num_scales - 1) * (
    #             args.sigma_min ** (1 / args.rho) - args.sigma_max ** (1 / args.rho)
    #         )
    #         t = t**args.rho
    #         x_t = image + noise * append_dims(t, dims)
    #         model_kwargs = dict(y=None)
    #         _, x0 = diffusion.denoise(model, x_t, t, **model_kwargs)
    #         # x0 = x0.clamp(-1, 1)
    #         if use_normalize:
    #             x0 = [vae.decode(x.unsqueeze(0)*std/0.5 + mean).sample for x in x0]
    #         else:
    #             x0 = [vae.decode(x.unsqueeze(0) / 0.18215).sample for x in x0]
    #         x0 = torch.cat(x0)
    #         torchvision.utils.save_image(x0, f"{test_dir}/x0_t={ind}.png", normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("flow-matching parameters")
    parser.add_argument("--generator",type=str,default="determ",help="type of seed generator",choices=["dummy", "determ", "determ-indiv"])
    parser.add_argument("--seed", type=int, default=42, help="seed used for initialization")
    parser.add_argument("--ckpt", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--ema", action="store_true", default=False)
    parser.add_argument("--test-interval", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=64, help="sample generating batch size")
    
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

    ###### sampling ######
    parser.add_argument("--cfg-scale", type=float, default=1.)
    parser.add_argument("--clip-denoised", action="store_true", default=True)
    parser.add_argument("--sampler", type=str, default='onestep')
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--s-tmin", type=float, default=0.0)
    parser.add_argument("--s-tmax", type=float, default=float("inf"))
    parser.add_argument("--s-noise", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--num-sampling", type=int, default=50000)
    parser.add_argument("--ts", type=str, default="0,22,39")
    
    ###### diffusion ######
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--weight-schedule", type=str, choices=["karras", "snr", "snr+1", "uniform", "truncated-snr", "ict"], default="uniform")
    parser.add_argument("--noise-sampler", type=str, choices=["uniform", "ict"], default="ict")
    parser.add_argument("--loss-norm", type=str, choices=["l1", "l2", "lpips", "huber", "adaptive"], default="huber")
    
    
    ###### dataset ######
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--num-in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--normalize-matrix", type=str, default=None)
    
    ###### compute fid ######
    parser.add_argument("--compute-fid", action="store_true", default=False, help="whether or not compute FID")
    parser.add_argument("--real-img-dir", default="./pytorch_fid/cifar10_train_stat.npy", help="directory to real images for FID computation")
    parser.add_argument("--output-log", type=str, default="fid.txt")
    
    args = parser.parse_args()
    main(args)
