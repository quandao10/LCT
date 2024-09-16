python pytorch_fid/fid_score.py \
    --batch-size 32 \
    --device cuda:0 \
    --dims 2048 \
    generated_samples/latent_celeb256/explarge_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_ep0001400 \
    ~/repo/edm2/data/real_samples/celeba_256

python ~/envs/slack_workflow/running_finished.py
