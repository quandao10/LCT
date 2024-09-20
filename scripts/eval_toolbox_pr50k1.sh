export CUDA_VISIBLE_DEVICES="0"

for epoch in 1400 1375 1350 1325 1300 1225
do
    python eval_toolbox/calc_metrics.py \
        --metrics pr50k3_full \
        --data ~/datasets/celeba_256_jpg \
        --mirror 1 \
        --gen_data generated_samples/latent_celeb256/explarge_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_cauchynew_newc_ep000${epoch} \
        --img_resolution 256
done

python ~/envs/slack_workflow/running_finished.py
