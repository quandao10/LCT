import numpy as np
import lmdb
import os
import io
from glob import glob
import torch
import torch.utils.data as data


class LatentDataset(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.train = train
        self.transform = transform
        if self.train:
            latent_paths = glob(f'{root}/train/*.npy')
        else:
            latent_paths = glob(f'{root}/val/*.npy')
        self.data = latent_paths

    def __getitem__(self, index):
        sample = np.load(self.data[index]).item()
        target = torch.from_numpy(sample["label"])
        x = torch.from_numpy(sample["input"])
        if self.transform is not None:
            x = self.transform(x)

        return x, target

    def __len__(self):
        return len(self.data)


class PreprocessedLatentDataset(data.Dataset):
    def __init__(self,
        path,
        use_labels,
    ):
        self._path = path
        self._use_labels = use_labels
        data = np.load(self._path, allow_pickle=True)
        self.mean = data.item()['mean']
        self.std = data.item()['std']
        if self._use_labels:
            self.label = data.item()['label']
        self.num_channels = 8
        self.has_labels = self._use_labels
        
    def __len__(self):
        return self.mean.shape[0]
    
    def __getitem__(self, index):
        latent_dist = np.concatenate([self.mean[index], self.std[index]], axis=0)
        if self._use_labels:
            label = self.label[index]
        else:
            label = np.array([])
        return latent_dist, label
    