MODE=$1
echo "Mode: $MODE"


resolution=256

if [ $MODE -eq 1 ]; then
    # # 1. Convert raw ImageNet data to a ZIP archive at 256x256 resolution
    python -m dataset_preprocessing.dataset_tools convert \
        --source=/research/cbim/archive/Datasets/ImageNet/ILSVRC2012_train/data \
        --dest=/common/users/qd66/dataset/real_imagenet_${resolution}/images \
        --resolution=${resolution}x${resolution} --transform=center-crop-dhariwal

elif [ $MODE -eq 2 ]; then
    # 2. Convert the pixel data to VAE latents
    python -m dataset_preprocessing.dataset_tools encode \
        --source=/common/users/qd66/dataset/real_imagenet_${resolution}/images \
        --dest=/common/users/qd66/dataset/real_imagenet_${resolution}/vae-sdvae-ft-ema
fi


# if [ $MODE -eq 1 ]; then
#     # # 1. Convert raw ImageNet data to a ZIP archive at 256x256 resolution
#     python dataset_preprocessing/dataset_tools.py convert \
#         --source=/research/cbim/archive/Datasets/ImageNet/ILSVRC2012_train/data \
#         --dest=/lustre/scratch/client/movian/research/groups/anhgroup/anhnd72/datasets/LCT/real_imagenet_${resolution}/images \
#         --resolution=${resolution}x${resolution} --transform=center-crop-dhariwal

# elif [ $MODE -eq 2 ]; then
#     # 2. Convert the pixel data to VAE latents
#     python dataset_preprocessing/dataset_tools.py encode \
#         --source=/lustre/scratch/client/movian/research/groups/anhgroup/anhnd72/datasets/LCT/real_imagenet_${resolution}/images \
#         --dest=/lustre/scratch/client/movian/research/groups/anhgroup/anhnd72/datasets/LCT/real_imagenet_${resolution}/vae-sdvae-ft-ema
# fi