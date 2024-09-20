export CUDA_VISIBLE_DEVICES="0"

epoch=0775

python eval_toolbox/calc_metrics.py \
    --metrics pr50k3_full \
    --data ~/datasets/celeba_256_jpg \
    --mirror 1 \
    --gen_data generated_samples/latent_celeb256/explarge_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_ep000${epoch} \
    --img_resolution 256

python eval_toolbox/calc_metrics.py \
    --metrics pr50k3_full \
    --data ~/datasets/celeba_256_jpg \
    --mirror 1 \
    --gen_data generated_samples/latent_celeb256/explarge_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_cauchynew_ep000${epoch} \
    --img_resolution 256

python eval_toolbox/calc_metrics.py \
    --metrics pr50k3_full \
    --data ~/datasets/celeba_256_jpg \
    --mirror 1 \
    --gen_data generated_samples/latent_celeb256/explarge_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_cauchynew_newc_ep000${epoch} \
    --img_resolution 256

python ~/envs/slack_workflow/running_finished.py
