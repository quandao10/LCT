import torch
from datasets_prep import get_dataset
from argparse import Namespace
import numpy as np

feats = []
dataset = get_dataset(Namespace(**dict(dataset="latent_celeb256")))
for i in range(len(dataset)):
    feat = dataset[i][0] / 0.18215
    feats.append(feat)
feats = torch.stack(feats)
std, mean = torch.std_mean(feats, dim=0)
np.save("celeb256_stat.npy", {"mean": mean, "std": std})
print(mean.shape, std.shape)
