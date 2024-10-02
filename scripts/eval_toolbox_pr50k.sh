export CUDA_VISIBLE_DEVICES="0"

python eval_toolbox/calc_metrics.py \
    --metrics fid50k_full,pr50k3_full \
    --data ~/datasets/celeba_256_jpg \
    --mirror 1 \
    --gen_data generated_samples/latent_celeb256/expdhariwal_unet_ict_large_batchsize_lr_decay_1k6_epoch_normalize_0.5_1280_nogradnorm_cauchy_positional_ep0001375 \
    --img_resolution 256

python ~/envs/slack_workflow/running_finished.py
