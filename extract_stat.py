import torch
from datasets_prep import get_dataset
from argparse import Namespace
import numpy as np
from tqdm import tqdm

feats = []
dataset = get_dataset(Namespace(**dict(dataset="latent_imagenet256")))
for i in tqdm(range(len(dataset))):
    feat = dataset[i][0]/0.18215
    feats.append(feat)
feats = torch.stack(feats)
std, mean = torch.std_mean(feats, dim=0)
np.save("latent_imagenet256_stat.npy", {"mean": mean, "std": std})
print(mean.shape, std.shape)

