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
from torchvision.utils import make_grid, save_image



def precompute_ssl_feat(args):
    device = torch.device("cuda")
    encoders, encoder_types, architectures = load_encoders(args.enc_type, device)
    args.get_file_id = True
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
            raw_image = (raw_image * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            # raw_image.shape = [64, 3, 256, 256] --> create a grid of 8x8 images, then save it
            grid = make_grid(raw_image, nrow=8, padding=2)
            save_image(grid, os.path.join(save_img_dir, f"{i}.png"))
            print(f"Saved: {os.path.join(save_img_dir, f'{i}.png')}")
            import ipdb; ipdb.set_trace()
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    precompute_ssl_feat(args)