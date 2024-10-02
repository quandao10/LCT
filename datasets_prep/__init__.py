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
        # dataset = CustomDataset("celebhq256", "/research/cbim/vast/qd66/workspace/dataset/vim/dataset/latent_celeba_256")
        dataset = CustomDataset("celebhq256", "./dataset/latent_celeba_256")
    elif args.dataset == "latent_ffhq256":
        dataset = CustomDataset("ffhq", "./dataset/latent_ffhq_256")
    # elif args.dataset == "latent_ffhq256":
    #     dataset = CustomDataset("ffhq", "features/ffhq256_features")
    # elif args.dataset == "latent_ffhq256_flip":
    #     dataset = CustomDataset("ffhq_flip", "features/ffhq256_feature_flip")
    # elif args.dataset == "latent_church256":
    #     dataset = CustomDataset("church256", "features/church256_features")
    # elif args.dataset == "latent_church256_flip":
    #     dataset = CustomDataset("church256_flip", "features/lsun_flip")
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
