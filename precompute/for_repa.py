"""
Input:
    - VAE
    - Dataset
    - Repa encoder

Output:
    - Latent 
    - SSL features
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
from datasets_prep import get_dataset
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from repa_utils import preprocess_raw_image
from repa_utils import load_encoders
from PIL import Image
from torchvision.utils import save_image
import numpy as np
from torchvision import transforms
import PIL
from torch.utils.data import Dataset
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
import multiprocessing
from functools import partial


CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type):
    resolution = x.shape[-1]
    if "clip" in enc_type:
        x = x / 255.0
        x = torch.nn.functional.interpolate(
            x, 224 * (resolution // 256), mode="bicubic"
        )
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif "mocov3" in enc_type or "mae" in enc_type:
        x = x / 255.0
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif "dinov2" in enc_type:
        x = x / 255.0
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(
            x, 224 * (resolution // 256), mode="bicubic"
        )
    elif "dinov1" in enc_type:
        x = x / 255.0
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif "jepa" in enc_type:
        x = x / 255.0
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(
            x, 224 * (resolution // 256), mode="bicubic"
        )

    return x


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


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
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


class REPADataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.img_list = [os.path.join(path, img) for img in os.listdir(path)]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image_name = os.path.basename(img_path).split(".")[0]

        # VAE image
        vae_image = Image.open(img_path).convert("RGB")
        vae_image = self.transform(vae_image)

        # REPA image
        repa_image = np.array(PIL.Image.open(img_path))
        repa_image = repa_image.reshape(*repa_image.shape[:2], -1).transpose(2, 0, 1)
        repa_image = torch.from_numpy(repa_image).float()

        return vae_image, repa_image, image_name


def save_features_func(save_tuple):
    image_name, ssl_feat, latent, ssl_feat_dir, vae_dir = save_tuple
    if ssl_feat is not None:
        np.save(f"{ssl_feat_dir}/{str(image_name)}.npy", ssl_feat)
    if latent is not None:
        np.save(f"{vae_dir}/{str(image_name)}.npy", latent)


@torch.no_grad()
def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    print(f"\033[33mProcessing {args.image_dir}...\033[0m")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\033[33mRunning VAE: {args.run_VAE}\033[0m")
    print(f"\033[33mRunning SSL: {args.run_SSL}\033[0m ({args.SSL_model})")
    assert args.run_VAE or args.run_SSL, f"You must run at least one of VAE {args.run_VAE} or SSL {args.run_SSL}"
    
    ssl_feat_dir = None
    vae_dir = None
    
    # VAE
    if args.run_VAE:
        vae_dir = os.path.join(args.output_dir, "vae")
        os.makedirs(vae_dir, exist_ok=True)
        print(f"\033[33mVAE output dir: {vae_dir}\033[0m")

        vae = AutoencoderKL.from_pretrained(args.vae_type).to(device)
        vae.requires_grad_(False)

    # SSL
    if args.run_SSL:
        ssl_feat_dir = os.path.join(args.output_dir, f"ssl_feat_{args.SSL_model}")
        os.makedirs(ssl_feat_dir, exist_ok=True)
        print(f"\033[33mSSL features output dir: {ssl_feat_dir}\033[0m")

        encoders, encoder_types, architectures = load_encoders(args.repa_enc_type, device)
        assert len(encoders) == 1
        ssl_encoder = encoders[0]
        encoder_type = encoder_types[0]

    do = input("Do you want to continue? (y/n)")
    if do == "n":
        exit()
    elif do == "y":
        pass
    else:
        raise ValueError("Invalid input")
    
    # Dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    dataset = REPADataset(args.image_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    for i, (vae_image, repa_image, image_name) in enumerate(tqdm(loader)):
        # VAE latent
        if args.run_VAE:
            vae_image = vae_image.to(device)
            with torch.no_grad():
                latent = vae.encode(vae_image).latent_dist.sample().mul_(0.18215)
            latent = latent.detach().cpu().numpy()
        else:
            latent = [None] * len(image_name)
        
        # SSL features
        if args.run_SSL:
            repa_image = repa_image.to(device)

            raw_image_ = preprocess_raw_image(repa_image, encoder_type)
            z = ssl_encoder.forward_features(raw_image_)
            if "mocov3" in encoder_type:
                ssl_feat = z = z[:, 1:]
            if "dinov2" in encoder_type:
                ssl_feat = z["x_norm_patchtokens"]
            ssl_feat = ssl_feat.detach().cpu().numpy()
        else:
            ssl_feat = [None] * len(image_name)

        # Prepare data for parallel saving
        save_tuples = [
            (name, feat, lat, ssl_feat_dir, vae_dir)
            for name, feat, lat in zip(image_name, ssl_feat, latent)
        ]

        # Use process pool to save features in parallel
        with multiprocessing.Pool(
            processes=min(args.num_workers, len(save_tuples))
        ) as pool:
            pool.map(save_features_func, save_tuples)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", type=str, default="/home/khanhdn10/repo/lct/dataset/"
    )
    parser.add_argument("--dataset_name", type=str, default="latent_celeb256")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--repa_enc_type", type=str, default="dinov2-vit-b")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--vae_type", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument(
        "--SSL_model",
        type=str,
        default="dinov2-vit-b",
        choices=[
            "dinov2-vit-b",
            "dinov2-vit-l",
            "dinov2-vit-g",
            "dinov1-vit-b",
            "mocov3-vit-b",
            "mocov3-vit-l",
            "clip-vit-L",
            "jepa-vit-h",
            "mae-vit-l",
        ],
    )
    parser.add_argument("--run_VAE", type=bool, default=False)
    parser.add_argument("--run_SSL", type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
