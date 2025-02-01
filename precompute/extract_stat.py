import torch
from datasets_prep import RepaDataset
import numpy as np
import os

feats = []
base_dir = "/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/latent_celeb256"
print(f"Processing {base_dir}")
dataset = RepaDataset(base_dir)
for i in range(len(dataset)):
    feat = dataset[i][0] / 0.18215
    feats.append(feat)
feats = torch.stack(feats)
std, mean = torch.std_mean(feats, dim=0)
np.save(os.path.join(base_dir, "celeb256_stat.npy"), {"mean": mean, "std": std})
print(f"mean: {mean}, std: {std}")
print(f"mean.shape: {mean.shape}, std.shape: {std.shape}")
print(f"Saved to {os.path.join(base_dir, 'celeb256_stat.npy')}")
