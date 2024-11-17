# build DC-AE models
# full DC-AE model list: https://huggingface.co/collections/mit-han-lab/dc-ae-670085b9400ad7197bb1009b
from efficientvit.efficientvit.ae_model_zoo import DCAE_HF
from torchvision.datasets import ImageFolder, ImageNet
import argparse
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from efficientvit.efficientvit.apps.utils.image import DMCrop
import os
import logging
from torch.utils.data import DataLoader, Dataset
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from tqdm import tqdm

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



def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    
    device = "cuda:0"
    torch.cuda.set_device(device)

    # Setup a feature folder:
    
    os.makedirs(args.features_path, exist_ok=True)
    os.makedirs(os.path.join(args.features_path, args.dataset), exist_ok=True)
    os.makedirs(os.path.join(args.features_path, args.dataset+"_res"), exist_ok=True)
    os.makedirs(os.path.join(args.features_path, args.dataset+'_label'), exist_ok=True)

    # Create model:
    vae = DCAE_HF.from_pretrained(f"mit-han-lab/dc-ae-f32c32-sana-1.0").to(device).eval()
    vae.requires_grad_(False)

    # Setup data:
    transform = transforms.Compose([
        DMCrop(args.image_size), # resolution
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # flip_transform = transforms.Compose([
    #     # transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(p=1),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])
    
    # for imagenet
    dataset = ImageFolder(args.data_path, transform=transform)
    # flip_dataset = ImageFolder(args.data_path, transform=flip_transform)
    
    # for other dataset
    # dataset = CustomDataset(args.data_path, transform=transform)
    # flip_dataset = CustomDataset(args.data_path, transform=flip_transform)
    
    loader = DataLoader(
        dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    # flip_loader = loader = DataLoader(
    #     flip_dataset,
    #     batch_size = 1,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     drop_last=False
    # )
    
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            z = vae.encode(x)
            if i < 1000:
                x_res = vae.decode(z)
        z = z.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        np.save(f'{args.features_path}/{args.dataset}/{str(i).zfill(9)}.npy', z[0])
        np.save(f'{args.features_path}/{args.dataset}_label/{str(i).zfill(9)}.npy', y[0])
        if i < 1000:
            save_image(x_res * 0.5 + 0.5, f'{args.features_path}/{args.dataset}_res/{str(i).zfill(9)}.jpg')
        
    # print("save flip loader")
    # N = len(dataset)
    # print("number of image sample: {}".format(N))
    # for i, x in enumerate(tqdm(flip_loader)):
    #     x = x.to(device)
    #     # y = y.to(device)
    #     with torch.no_grad():
    #         # Map input images to latent space + normalize latents:
    #         x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    #     x = x.detach().cpu().numpy()
    #     # y = y.detach().cpu().numpy()
    #     np.save(f'{args.features_path}/ffhq256_feature_flip/{str(i+N).zfill(9)}.npy', x[0])
    #     # np.save(f'{args.features_path}/imagenet256_label_flip/{str(i+N).zfill(9)}.npy', y[0])

if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--features-path", type=str, default="../dataset/features_dcvae")
    parser.add_argument("--dataset", type=str, default="celeba256")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=50)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    main(args)