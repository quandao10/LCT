# find all files in the directory
import os

directory = "/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/real_imagenet_256/images"
for root, dirs, files in os.walk(directory):
    for file in files:
        print(os.path.join(root, file))