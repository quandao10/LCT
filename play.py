# find all files in the directory
import os

directory = "/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/real_imagenet_256/images"
all_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        all_files.append(os.path.join(root, file))

print(f"\033[33mFound {len(all_files)} files\033[0m")

import ipdb; ipdb.set_trace()