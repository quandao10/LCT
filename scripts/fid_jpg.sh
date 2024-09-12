export CUDA_VISIBLE_DEVICES="0"

python pytorch_fid/fid_score.py \
    --batch-size 32 \
    --device cuda:0 \
    --dims 2048 \
    generated_samples/latent_celeb256/expdhariwal_unet_ict_large_batchsize_lr_decay_1k_epoch_normalize_0.5_160_nogradnorm_cauchy_positional_ep0001000 \
    ~/datasets/celeba_256_jpg

for epoch in 1175 1375 1400 1600
do
    python pytorch_fid/fid_score.py \
        --batch-size 32 \
        --device cuda:0 \
        --dims 2048 \
        generated_samples/latent_celeb256/expdhariwal_unet_ict_large_batchsize_lr_decay_1k6_epoch_normalize_0.5_1280_nogradnorm_cauchy_positional_ep000${epoch} \
        ~/datasets/celeba_256_jpg
done

python ~/envs/slack_workflow/running_finished.py
