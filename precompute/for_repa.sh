image_dir="/lustre/scratch/client/movian/research/users/khanhdn10/datasets/celeba_256_png"
dataset_name="celeba_256"
output_dir="/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/latent_celeb256"
repa_enc_type="dinov2-vit-b"
batch_size=128
num_workers=32
vae_type="stabilityai/sd-vae-ft-ema"

python -m precompute.for_repa --image_dir $image_dir --dataset_name $dataset_name --output_dir $output_dir --repa_enc_type $repa_enc_type --batch_size $batch_size --num_workers $num_workers --vae_type $vae_type