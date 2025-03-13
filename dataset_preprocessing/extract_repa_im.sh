image_dir="/common/users/qd66/dataset/real_imagenet_256/images/"
dataset_name="real_imagenet_256"
output_dir="/common/users/qd66/repa/"
SSL_model="dinov2-vit-b" # clip-vit-L, dinov2-vit-b, dinov2-vit-l, mocov3-vit-b, mocov3-vit-l, jepa-vit-h, mae-vit-l
batch_size=128
num_workers=64 # 64
vae_type="stabilityai/sd-vae-ft-ema"
run_VAE=False
run_SSL=True
is_ImageNet=True

python -m dataset_preprocessing.precompute_ssl_repa_im --image_dir $image_dir \
                                                        --dataset_name $dataset_name \
                                                        --output_dir $output_dir \
                                                        --batch_size $batch_size \
                                                        --num_workers $num_workers \
                                                        --vae_type $vae_type \
                                                        --run_VAE $run_VAE \
                                                        --run_SSL $run_SSL \
                                                        --SSL_model $SSL_model \
                                                        --is_ImageNet $is_ImageNet \

# python preprocessing_imagenet/precompute_ssl_repa.py --image_dir $image_dir --dataset_name $dataset_name --output_dir $output_dir --batch_size $batch_size --num_workers $num_workers --vae_type $vae_type --run_VAE $run_VAE --run_SSL $run_SSL --SSL_model $SSL_model --is_ImageNet $is_ImageNet

# 00265/img00265542.png