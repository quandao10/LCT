# write a script to read json file get the image path and copy to a new folder  
import json
import os
import shutil
from tqdm import tqdm
from PIL import Image


all_paths = []
# read json file
with open('statistic/imagenet25_class_to_images.json', 'r') as f:
    data = json.load(f)
    for key, value in data.items():
        all_paths.extend(value)

# create new folder
# os.makedirs('~/workspace/dataset/imagenet25', exist_ok=True)

# copy image to new folder
for item in tqdm(all_paths):
    # convert image from png to jpg then copy to new folder
    image_path = os.path.join("/common/users/qd66/dataset/real_imagenet_256/images/", item)
    image = Image.open(image_path)
    image.save(os.path.join('/research/cbim/vast/qd66/workspace/dataset/imagenet25/', item.split('/')[-1].replace('.png', '.jpg')))
    