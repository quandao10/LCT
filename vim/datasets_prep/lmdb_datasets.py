# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image


def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    elif dataset == 'lsun_church':
        return 126227
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


class LMDBDataset(data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            lmdb_path = os.path.join(root, 'train.lmdb')
        else:
            lmdb_path = os.path.join(root, 'validation.lmdb')
        self.lmdb_path = lmdb_path
        # self.env = lmdb.open(lmdb_path, readonly=True, max_readers=1,
        #                            lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

        # compute size
        self.open_lmdb()
        with self.env.begin() as txn:
            self.size = txn.stat()['entries']
        self.env.close()
        del self.env
    
    def open_lmdb(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        # self.txn = self.env.begin(write=False, buffers=True)

    def __getitem__(self, index):
        target = [0]
        if not hasattr(self, 'env'):
            self.open_lmdb()
        with self.env.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB')
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.size
