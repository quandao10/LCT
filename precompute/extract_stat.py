import torch
from datasets_prep import RepaDataset
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

base_dir = "/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/latent_celeb256"
print(f"Processing {base_dir}")

# Create dataset and dataloader for parallel processing
dataset = RepaDataset(base_dir)
loader = DataLoader(dataset, batch_size=128, num_workers=32, shuffle=False, drop_last=False)

# Pre-allocate tensor to store all features
total_samples = len(dataset)
feat_ls = []

import ipdb; ipdb.set_trace()
# Load features in batches
for batch in tqdm(loader):
    feat_ls.append(batch[0] / 0.18215)
    import ipdb; ipdb.set_trace()

# Calculate statistics
feats = torch.stack(feat_ls)
std, mean = torch.std_mean(feats, dim=0)

# Save results
np.save(os.path.join(base_dir, "celeb256_stat.npy"), {"mean": mean.cpu().numpy(), "std": std.cpu().numpy()})
print(f"mean: {mean}, std: {std}")
print(f"mean.shape: {mean.shape}, std.shape: {std.shape}")
print(f"Saved to {os.path.join(base_dir, 'celeb256_stat.npy')}")
print(f"Total samples: {total_samples}")