MODE=$1
echo "Mode: $MODE"


resolution=256

if [ $MODE -eq 1 ]; then
    # # 1. Convert raw ImageNet data to a ZIP archive at 256x256 resolution
    python -m dataset_preprocessing.dataset_tools convert \
        --source=/research/cbim/archive/Datasets/imagenet/ILSVRC2012_train/data \
        --dest=/common/users/qd66/dataset/real_imagenet_${resolution}/images \
        --resolution=${resolution}x${resolution} --transform=center-crop-dhariwal

elif [ $MODE -eq 2 ]; then
    # 2. Convert the pixel data to VAE latents
    python -m dataset_preprocessing.dataset_tools encode \
        --model-url=zelaki/eq-vae \
        --source=/common/users/qd66/dataset/real_imagenet_${resolution}/images \
        --dest=/research/cbim/vast/qd66/workspace/dataset/repa/latent_imagenet256/eq_vae/
elif [ $MODE -eq 3 ]; then
    # 3. Convert the pixel data to VAE latents with json
    python preprocessing_imagenet/dataset_tools.py encodewithjson \
        --model-url=zelaki/eq-vae \
        --source=/common/users/qd66/dataset/real_imagenet_${resolution}/images \
        --dest=/common/users/qd66/dataset/real_imagenet_${resolution}/vae-eqvae-ft-ema-subset \
        --meta-data=statistic/imagenet25_class_to_images.json
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