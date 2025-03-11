import torch
import torchvision.transforms as transforms
from datasets_prep.data_transforms import center_crop_arr
from datasets_prep.inpainting_dataset import InpaintingTrainDataset
from datasets_prep.lmdb_datasets import LMDBDataset
from datasets_prep.lsun import LSUN
from torchvision.datasets import CIFAR10, ImageNet
import numpy as np
from torch.utils.data import Dataset
import os
from models.script_util import parse_repa_enc_info, parse_repa_enc_info_v2
from tqdm import tqdm
import json


class CustomDataset(Dataset):
    def __init__(self, dataset, features_dir, labels_dir=None):
        self.features_dir = features_dir
        self.labels_dir = labels_dir
        self.dataset = dataset

    def __len__(self):
        if self.dataset == "imagenet":
            return 1281167
        elif self.dataset == "celebhq":
            return 30000
        elif self.dataset == "celebhq_flip":
            return 60000
        elif self.dataset == "celebhq256":
            return 27000
        elif self.dataset == "church256":
            return 120000
        elif self.dataset == "church256_flip":
            return 240000
        elif self.dataset == "ffhq_flip":
            return 140000
        elif self.dataset == "ffhq":
            return 70000

    def __getitem__(self, idx):
        file_id = f"{str(idx).zfill(9)}.npy"
        features = np.load(os.path.join(self.features_dir, file_id))
        if self.labels_dir is not None:
            labels = np.load(os.path.join(self.labels_dir, file_id))
        else:
            return torch.from_numpy(features), torch.tensor(0)
        return torch.from_numpy(features.copy()), torch.from_numpy(labels)


class RepaDataset(Dataset):
    def __init__(self, base_dir, repa_enc_info):
        """
        base_dir: Base directory containing the dataset
        repa_enc_info: Semicolon-separated encoder info, e.g. "4:dinov2-vit-b;6:dinov2-vit-b"
        """
        # Parse encoder info into dictionaries
        _, encoder, _ = parse_repa_enc_info_v2(repa_enc_info)
        
        # Get unique encoder types
        self.encoder = encoder
        
        # Create SSL feature directories mapping {encoder_type: directory}
        self.ssl_feat_dir = os.path.join(base_dir, f"ssl_feat_{self.encoder}") 
        self.vae_dir = os.path.join(base_dir, "vae")
        
        # Print loading info
        print(f"\033[1;35mLoading ssl_feat_dir: {self.ssl_feat_dir}\033[0m")
        print(f"\033[1;35mLoading vae_dir: {self.vae_dir}\033[0m")

        # Get list of files (use first SSL dir as reference)
        print("\033[1;35mScanning directory for files...\033[0m")
        all_files = os.listdir(self.ssl_feat_dir)
        self.basename_files = [os.path.basename(file).split(".")[0] for file in tqdm(all_files)]

    def __len__(self):
        return len(self.basename_files)

    def __getitem__(self, idx):
        file_id = self.basename_files[idx]
        
        # VAE latent
        latent = np.load(os.path.join(self.vae_dir, f"{str(file_id).zfill(9)}.npy"))
        latent = torch.from_numpy(latent)

        # Load SSL features and organize by encoder type
        ssl_feat = np.load(os.path.join(self.ssl_feat_dir, f"{file_id}.npy"))
        ssl_feat = torch.from_numpy(ssl_feat)
        
        # label
        label = torch.tensor(0)
        return latent, ssl_feat, label



class ImageNet_dataset(torch.utils.data.Dataset):
    def __init__(self, base_dir, use_labels=True, repa_enc_info=None):
        # VAE
        self.base_dir = base_dir
        self.vae_path = os.path.join(base_dir, "vae_")
        self.use_labels = use_labels
        with open("/research/cbim/vast/qd66/workspace/LCT/statistic/imagenet25_class_to_images.json") as f:
            self.items = json.load(f)
        self.class_indices = list(self.items.keys())
        self.num_classes = len(self.class_indices)
        # sort classes, then get the index of each class
        self.class_indices.sort()
        self.classidx_to_label = {self.class_indices[i]: i for i in range(self.num_classes)}
        self.label_to_classidx = {v: k for k, v in self.classidx_to_label.items()}

        self.list_of_files = []
        for classidx, list_img in self.items.items():
            for img in list_img:
                img = img.replace('png', 'npy').replace('img', 'img-mean-std-')
                self.list_of_files.append((img, classidx))

        # REPA
        self.repa_enc_info = repa_enc_info
    
        # Parse encoder info into dictionaries
        depth, enc, z_dim = parse_repa_enc_info_v2(repa_enc_info)
        
        # Create SSL feature directories mapping {encoder_type: directory}
        self.ssl_feat_dir = os.path.join(self.base_dir, f"ssl_feat_{enc}") 
        

    def __len__(self):
        return len(self.list_of_files)
    
    def __getitem__(self, index):
        # VAE latent and label
        npy_file, classidx = self.list_of_files[index]
        # npy_file: 00265/img-mean-std-00265542.npy
        label = self.classidx_to_label[classidx]
        label = torch.tensor(int(label))
        npy_data = np.load(os.path.join(self.vae_path, npy_file))
        mean, std = np.array_split(npy_data, indices_or_sections=2, axis=0) 
        mean, std = torch.from_numpy(mean), torch.from_numpy(std)   
        sample = mean + torch.randn_like(mean) * std
        # SSL features
        # Load SSL features and organize by encoder type
        # img00265542.png.npy
        ssl_file = ("img" + npy_file.split("-")[-1]).replace(".npy", ".png.npy")
        ssl_feat = np.load(os.path.join(self.ssl_feat_dir, ssl_file))
        ssl_feat = torch.from_numpy(ssl_feat)
        return sample, ssl_feat, label

def get_repa_dataset(args):   
    if args.dataset == "subset_imagenet_256":
        dataset = ImageNet_dataset(args.datadir, use_labels=True, repa_enc_info=args.repa_enc_info)
        return dataset
    else:
        return RepaDataset(args.datadir, args.repa_enc_info)

def get_dataset(args):
    if args.dataset == "cifar10":
        dataset = CIFAR10(
            args.datadir,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            download = True
        )

    elif args.dataset == "imagenet_256":
        dataset = ImageNet(
            args.datadir,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )

    elif args.dataset == "lsun_church":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_data = LSUN(root=args.datadir, classes=["church_outdoor_train"], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)

    elif args.dataset == "lsun_bedroom":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.CenterCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_data = LSUN(root=args.datadir, classes=["bedroom_train"], transform=train_transform)
        subset = list(range(0, 120000))
        dataset = torch.utils.data.Subset(train_data, subset)

    elif args.dataset == "celeba_256":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = LMDBDataset(root=args.datadir, name="celeba", train=True, transform=train_transform)

    elif args.dataset == "celeba_512":
        from torchtoolbox.data import ImageLMDB

        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ImageLMDB(db_path=args.datadir, db_name="celeba_512", transform=train_transform, backend="pil")

    elif args.dataset == "celeba_1024":
        from torchtoolbox.data import ImageLMDB

        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = ImageLMDB(db_path=args.datadir, db_name="celeba_1024", transform=train_transform, backend="pil")

    elif args.dataset == "ffhq_256":
        train_transform = transforms.Compose(
            [
                transforms.Resize(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = LMDBDataset(root=args.datadir, name="ffhq", train=True, transform=train_transform)
    elif args.dataset == "latent_celeb256":
        dataset = CustomDataset("celebhq256", "/common/users/qd66/dataset/latent_celeba_256")
        # dataset = CustomDataset("celebhq256", "./dataset/latent_celeba_256")
    elif args.dataset == "latent_ffhq256":
        dataset = CustomDataset("ffhq", "/research/cbim/vast/qd66/workspace/dataset/features/ffhq256_feature")
    # elif args.dataset == "latent_ffhq256_flip":
    #     dataset = CustomDataset("ffhq_flip", "features/ffhq256_feature_flip")
    # elif args.dataset == "latent_church256":
    #     dataset = CustomDataset("church256", "features/church256_features")
    elif args.dataset == "latent_church256_flip":
        dataset = CustomDataset("church256_flip", "/research/cbim/vast/qd66/workspace/dataset/features/lsun_flip")
    # elif args.dataset == "latent_celeb256":
    #     dataset = CustomDataset("celebhq256", "/research/cbim/vast/qd66/workspace/dataset/features/latent_celeba_256")
    # elif args.dataset == "latent_celeb512":
    #     dataset = CustomDataset("celebhq", "features/celebahq512_features")
    # elif args.dataset == "latent_celeb512_flip":
    #     dataset = CustomDataset("celebhq_flip", "features/celeb512_feature_flip")
    # elif args.dataset == "latent_celeb1024":
    #     dataset = CustomDataset("celebhq", "features/celebahq1024_features")
    # elif args.dataset == "latent_imagenet256":
    #     dataset = CustomDataset("imagenet", "features/imagenet256_features", "features/imagenet256_labels")
    # elif args.dataset == "latent_imagenet512":
    #     dataset = CustomDataset("imagenet", "features/imagenet512_features", "features/imagenet512_labels")
    return dataset


def get_inpainting_dataset(args):
    from datasets_prep.inpaint_preprocess.mask import get_mask_generator

    mask_gen = get_mask_generator(None, None)
    dataset = InpaintingTrainDataset(indir="dataset/data256x256/", mask_generator=mask_gen, transform=None)
    return dataset
