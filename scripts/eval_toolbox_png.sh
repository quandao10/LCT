export CUDA_VISIBLE_DEVICES="0"

# python eval_toolbox/calc_metrics.py \
#     --metrics fid50k_full,pr50k3_full \
#     --data ~/datasets/celeba_256_png \
#     --mirror 1 \
#     --gen_data generated_samples/latent_celeb256/expdhariwal_unet_ict_large_batchsize_lr_decay_1k_epoch_normalize_0.5_160_nogradnorm_cauchy_positional_ep0001000 \
#     --img_resolution 256

for epoch in 1175 1375 1400 1600
do
    python eval_toolbox/calc_metrics.py \
        --metrics fid50k_full,pr50k3_full \
        --data ~/datasets/celeba_256_png \
        --mirror 1 \
        --gen_data generated_samples/latent_celeb256/expdhariwal_unet_ict_large_batchsize_lr_decay_1k6_epoch_normalize_0.5_1280_nogradnorm_cauchy_positional_ep000${epoch} \
        --img_resolution 256
done

python ~/envs/slack_workflow/running_finished.py
