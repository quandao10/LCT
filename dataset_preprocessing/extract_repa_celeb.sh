set -x

image_dir="/research/cbim/vast/qd66/workspace/real_samples/celeba_256/"
dataset_name="celeba_256"
output_dir="/research/cbim/vast/qd66/workspace/dataset/repa/latent_celeb256/"
SSL_model="dinov2-vit-b" # "clip-vit-L" # clip-vit-L, dinov2-vit-b
batch_size=128
num_workers=64
vae_type="stabilityai/sd-vae-ft-ema"
run_VAE=False
run_SSL=True
is_ImageNet=False


python -m dataset_preprocessing.precompute_ssl_repa --image_dir $image_dir \
                                                --dataset_name $dataset_name \
                                                --output_dir $output_dir \
                                                --batch_size $batch_size \
                                                --num_workers $num_workers \
                                                --vae_type $vae_type \
                                                --run_VAE $run_VAE \
                                                --run_SSL $run_SSL \
                                                --SSL_model $SSL_model \
                                                --is_ImageNet $is_ImageNet \