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


class CustomDataset(Dataset):
    def __init__(self, path, transform):
        self.path = path
        self.img_list = [os.path.join(path, img) for img in os.listdir(path)]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def precompute_ssl_feat(args):
    device = torch.device("cuda")

    # DINOv2
    encoders, encoder_types, architectures = load_encoders(args.repa_enc_type, device)

    # Dataset
    args.get_file_id = True
    dataset = get_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    save_img_dir = os.path.join(args.output_dir, "debug_precompute_ssl_feat")
    os.makedirs(save_img_dir, exist_ok=True)
    print(f"\033[33mSaved: {save_img_dir}\033[0m")

    for i, (x, y, file_id) in enumerate(tqdm(loader)):
        x = x.to(device)

        ####################### REPA #######################
        ssl_feat = None
        with torch.no_grad():
            target = x.clone().detach()
            raw_image = target / vae.config.scaling_factor
            raw_image = vae.decode(raw_image.to(dtype=vae.dtype)).sample.float()
            save_image(
                raw_image, os.path.join(save_img_dir, f"{i}.png"), nrow=8, padding=2
            )
            import ipdb

            ipdb.set_trace()
            raw_image = (raw_image * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            ssl_feat = []
            # with torch.autocast(device_type='cuda', dtype=__dtype):
            with torch.autocast(device_type="cuda"):
                for encoder, encoder_type, arch in zip(
                    encoders, encoder_types, architectures
                ):
                    raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                    z = encoder.forward_features(raw_image_)
                    if "mocov3" in encoder_type:
                        z = z = z[:, 1:]
                    if "dinov2" in encoder_type:
                        z = z["x_norm_patchtokens"]
                    ssl_feat.append(z.detach())


def main(args):
    device = "cuda:0"
    torch.cuda.set_device(device)

    os.makedirs(args.features_path, exist_ok=True)
    os.makedirs(os.path.join(args.features_path, "celeba_256"), exist_ok=True)
    
    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    vae = AutoencoderKL.from_pretrained(args.vae_type).to(device)
    vae.requires_grad_(False)

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    dataset = CustomDataset(args.data_path, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    for i, x in enumerate(tqdm(loader)):
        x = x.to(device)
        # y = y.to(device)
        with torch.no_grad():
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        x = x.detach().cpu().numpy()
        # y = y.detach().cpu().numpy()

        np.save(f"{args.features_path}/celeba_256/{str(i).zfill(9)}.npy", x[0])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", type=str, default="/home/khanhdn10/repo/lct/dataset/"
    )
    parser.add_argument("--dataset_name", type=str, default="latent_celeb256")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--repa_enc_type", type=str, default="dinov2-vit-b")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--vae_type", type=str, default="stabilityai/sd-vae-ft-ema")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    precompute_ssl_feat(args)
