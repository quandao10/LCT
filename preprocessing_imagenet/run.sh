# # 1. Convert raw ImageNet data to a ZIP archive at 256x256 resolution
# python preprocessing_imagenet/dataset_tools.py convert \
#     --source=/lustre/scratch/client/vinai/users/anhnd72/datasets/real_imagenet/train \
#     --dest=/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/real_imagenet_256/images \
#     --resolution=256x256 --transform=center-crop-dhariwal

# 2. Convert the pixel data to VAE latents
python preprocessing_imagenet/dataset_tools.py encode \
    --source=/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/real_imagenet_256/images \
    --dest=/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/real_imagenet_256/vae-sdvae-ft-ema
