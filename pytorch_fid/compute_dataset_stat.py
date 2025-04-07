import argparse
import os

import numpy as np
from fid_score import compute_statistics_of_path
from inception import InceptionV3
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
class ImageDataset(Dataset):
    def __init__(self, datadir, transform=None):
        self.datadir = datadir
        self.transform = transform
        self.image_paths = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compute dataset stat")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--datadir", default="./data")
    parser.add_argument("--save_path", default="./pytorch_fid/cifar10_stat.npy")

    parser.add_argument("--image_size", type=int, default=32, help="size of image")
    parser.add_argument("--batch_size", type=int, default=100, help="size of image")
    parser.add_argument("--nz", type=int, default=2048, help="number of dimensions for fid")

    args = parser.parse_args()

    device = "cuda:0"

    # dataset = get_dataset(args)
    # dataset = ImageDataset(args.datadir, transform=transforms.ToTensor())
    # save_dir = "./real_samples/{}/".format(args.dataset)
    # os.makedirs(save_dir, exist_ok=True)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=4,  # cpu_count(),
    # )
    # for i, (x, _) in enumerate(tqdm(dataloader)):
    #     x = x.to(device, non_blocking=True)
    #     x = (x + 1.0) / 2.0  # move to 0 - 1
    #     assert 0 <= x.min() and x.max() <= 1
    #     for j, x in enumerate(x):
    #         index = i * args.batch_size + j
    #         torchvision.utils.save_image(x, "{}/{}.jpg".format(save_dir, index))
    #     print("Generate batch {}".format(i))
    # print("Save images in {}".format(save_dir))

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.nz]

    model = InceptionV3([block_idx]).to(device)
    mu, sigma = compute_statistics_of_path(args.datadir, model, batch_size=100, dims=args.nz, device=device, resize=0)
    print(mu.shape, sigma.shape)

    save_path = args.save_path
    save_dict = {"mu": mu, "sigma": sigma}
    with open(save_path, "wb") as f:
        np.save(f, save_dict, allow_pickle=True)

    # test
    if save_path.endswith(".npz") or save_path.endswith(".npy"):
        f = np.load(save_path, allow_pickle=True)
        try:
            m, s = f["mu"][:], f["sigma"][:]
        except IndexError:
            m, s = f.item()["mu"][:], f.item()["sigma"][:]
        print(m.shape, s.shape)
