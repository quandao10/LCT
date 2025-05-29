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
from models.network_udit import UDiT_models
from models.script_util import (
    create_model_and_diffusion,
)
from models.karras_diffusion import karras_sample
import numpy as np
from tqdm import tqdm
from tokenizer.vavae import VA_VAE
from calflops import calculate_flops

def decode_image(vae, latent, vae_name):
    latent = latent.unsqueeze(0)
    if vae_name == "vae" or vae_name == "eq_vae":
        return vae.decode(latent).sample
    elif vae_name == "va_vae":
        return vae.decode_to_images(latent)
    else:
        raise ValueError(f"Invalid vae name: {vae_name}")

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True  # True: fast but may lead to some small numerical differences
    torch.set_grad_enabled(False)
    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
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
    elif args.dataset in ["latent_imagenet_256", "imagenet_256", "imagenet_256_va"]:
        real_img_dir = "pytorch_fid/imagenet_stat.npy"
    elif args.dataset == "subset_imagenet_256":
        real_img_dir = "pytorch_fid/imagenet25.npy"
    else:
        real_img_dir = args.real_img_dir

    to_range_0_1 = lambda x: (x + 1.0) / 2.0

    model, diffusion = create_model_and_diffusion(args)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}") 
    # x = torch.randn(1, args.num_in_channels, args.image_size, args.image_size).to(device)
    # t = torch.zeros(1).to(device)
    # c = torch.zeros(1).to(device, dtype=torch.int) 
    # flops, macs, params = calculate_flops(
    #     model=model, kwargs={"x": x, "t": t, "y": c}, output_as_string=False, output_precision=4
    # )
    # print("GFLOPs:%.2f Params:%.2fM \n" % (flops / 2 / 10**9, params / 10**6))
    # exit(0)
    model.to(device=device)
    if args.vae == "vae":
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    elif args.vae == "eq_vae":
        vae = AutoencoderKL.from_pretrained(f"zelaki/eq-vae").to(device)
    elif args.vae == "va_vae":
        vae = VA_VAE(
            f'tokenizer/configs/vavae_f16d32.yaml',
        )
    ckpt = torch.load(args.ckpt, map_location=torch.device(f'cuda:{device}'))
    print("Finish loading model")
    # loading weights from ddp in single gpu
    if not args.ema:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt["ema"], strict=False)
    model.eval()
    
    del ckpt
    args.exp = args.ckpt.split("/")[-3]
    args.epoch_id = args.ckpt.split("/")[-1][:-3]
    save_dir = "./generated_samples/{}/exp{}_ep{}".format(args.dataset, args.exp, args.epoch_id)
    
    if args.cfg_scale > 1.0:
        save_dir += "_cfg{}".format(args.cfg_scale)
    if not args.ema:
        save_dir += "_noEMA"
    if rank == 0 and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # seed generator
    # seed should be aligned with rank
    generator = get_generator(args.generator, args.num_sampling, seed)
    use_label = True if "imagenet" in args.dataset else False
    use_normalize = args.normalize_matrix is not None
    if use_normalize:
        if args.normalize_matrix.endswith(".npy"):
            data = np.load(args.normalize_matrix, allow_pickle=True).item()
        elif args.normalize_matrix.endswith(".pt"):
            data = torch.load(args.normalize_matrix)
        else:
            raise ValueError(f"Invalid normalize matrix file: {args.normalize_matrix}")
        try:
            mean = data["mean"].to(device)
            std = data["std"].to(device)
        except:
            mean = torch.tensor(data["mean"]).to(device)
            std = torch.tensor(data["std"]).to(device)
        
    if "imagenet" in args.dataset and args.use_karras_normalization:
        print("using karras normalization for imagenet")
        mean = torch.tensor([5.81, 3.25, 0.12, -2.15]).view(1, -1, 1, 1).to(device)
        std =  torch.tensor([4.17, 4.62, 3.71, 3.28]).view(1, -1, 1, 1).to(device)
    
    # from models.nn import append_zero
    # # import numpy as np
    # def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    #     """Constructs the noise schedule of Karras et al. (2022)."""
    #     ramp = torch.linspace(0, 1, n)
    #     min_inv_rho = sigma_min ** (1 / rho)
    #     max_inv_rho = sigma_max ** (1 / rho)
    #     sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    #     return append_zero(sigmas).to(device)
    
    # sigmas = get_sigmas_karras(512, args.sigma_min, args.sigma_max, device=device)
    # print(sigmas)
    # # print(sigmas.shape)
    # time_data = 1000 * 0.25 * torch.log(sigmas + 1e-44)
    # # print(time_data.shape)
    # time_embed = model.t_embedder
    # time_mlp = model.cond_mixing_mlp
    # lambda_mlp = time_mlp(time_embed(time_data))
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(6, 5))
    # plt.plot(-2*np.log(sigmas.detach().cpu().numpy()), lambda_mlp.detach().cpu().numpy())
    # plt.xlabel("snr(t)")
    # plt.ylabel("λ")
    # plt.tight_layout()
    # # plt.title("λ vs snr(t)")
    # plt.savefig("lambda_mlp.png")
    # exit(0)

    def run_sampling(num_samples, generator):
        noise = generator.randn(num_samples, args.num_in_channels, args.image_size, args.image_size).to(device)*args.sigma_max if args.fwd == "ve" else \
            generator.randn(num_samples, args.num_in_channels, args.image_size, args.image_size).to(device)
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
                        model.forward if (args.cfg_scale<=1.0) else model.forward_with_cfg,
                        (args.batch_size, args.num_in_channels, args.image_size, args.image_size),
                        steps=args.steps,
                        model_kwargs=model_kwargs,
                        device=device,
                        sampler=args.sampler,
                        sigma_min=args.sigma_min,
                        sigma_max=args.sigma_max,
                        s_churn=args.s_churn,
                        s_tmin=args.s_tmin,
                        s_tmax=args.s_tmax,
                        s_noise=args.s_noise,
                        noise=noise,
                        shift=args.shift,
                        ts=ts,
                    )
        if args.cfg_scale > 1.0:
            fake_sample, _ = fake_sample.chunk(2, dim=0)  
            # Remove null class samples       
        fake_sample = fake_sample*std/args.sigma_data + mean 
        fake_sample = [decode_image(vae, x, args.vae) for x in fake_sample]
        fake_image = [torch.clamp(x, -1, 1) for x in fake_sample]
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
        torchvision.utils.save_image(fake_image, f'{args.exp}_{args.epoch_id}{ema}.jpg', nrow=8, normalize=True, value_range=(-1, 1))
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("flow-matching parameters")
    parser.add_argument("--generator",type=str,default="determ",help="type of seed generator",choices=["dummy", "determ", "determ-indiv"])
    parser.add_argument("--seed", type=int, default=42, help="seed used for initialization")
    parser.add_argument("--ckpt", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--ema", action="store_true", default=False)
    parser.add_argument("--test-interval", action="store_true", default=False)
    parser.add_argument("--batch-size", type=int, default=64, help="sample generating batch size")
    parser.add_argument("--use-karras-normalization", action="store_true", default=False)
    
    ###### model ######
    parser.add_argument("--vae", type=str, choices=["vae", "eq_vae", "va_vae"], default="vae")  # Choice doesn't affect training
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
    parser.add_argument("--model-type", type=str, choices=["openai_unet", "song_unet", "dhariwal_unet"]+list(DiT_models.keys())+list(UDiT_models.keys()), default="openai_unet")
    parser.add_argument("--wo-norm", action="store_true", default=False)
    parser.add_argument("--linear-act", type=str, default=None)
    parser.add_argument("--norm-type", type=str, default="layer")
    parser.add_argument("--num-register", type=int, default=0)
    parser.add_argument("--separate-cond", action="store_true", default=False)
    parser.add_argument("--use-rope", action="store_true", default=False)
    parser.add_argument("--cond-mapping", action="store_true", default=False)
    parser.add_argument("--freq-type", type=str, default="none")
    parser.add_argument("--cond-mixing", action="store_true", default=False)
    parser.add_argument("--uw", action="store_true", default=False)
    
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
    parser.add_argument("--shift", type=int, default=0)
    ###### diffusion ######
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--weight-schedule", type=str, choices=["karras", "snr", "snr+1", "uniform", "truncated-snr", "ict"], default="uniform")
    parser.add_argument("--noise-sampler", type=str, choices=["uniform", "ict"], default="ict")
    parser.add_argument("--loss-norm", type=str, choices=["l1", "l2", "lpips", "huber", "adaptive"], default="huber")
    parser.add_argument("--c-type", type=str, choices=["trig", "edm"], default="edm")
    parser.add_argument("--fwd", type=str, default="ve", choices=["vp", "ve", "flow", "cosin"])
    parser.add_argument("--p-mean", type=float, default=-0.4)
    parser.add_argument("--p-std", type=float, default=2.0)
    parser.add_argument("--rho", type=float, default=7)
    
    
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
    
    ###### REPA ######
    parser.add_argument("--use-repa", action="store_true", default=False)
    parser.add_argument("--repa-enc-info", type=str, default="4:dinov2-vit-b", help="Semicolon-separated encoder info. Format: 'encoder1;depth1;encoder2;depth2;...'")
    parser.add_argument("--projector-dim", type=int, default=2048)
    parser.add_argument("--repa-lamb", type=float, default=0.0)
    parser.add_argument("--z-dims", type=int, default=768)
    parser.add_argument("--repa-timesteps", type=str, default="full")
    parser.add_argument("--repa-relu-margin", type=float, default=0.5)
    parser.add_argument("--denoising-task-rate", type=float, default=0.25)
    parser.add_argument("--repa-mapper", type=str, default="repa")
    parser.add_argument("--mar-mapper-num-res-blocks", type=int, default=2)
    
    args = parser.parse_args()
    main(args)
