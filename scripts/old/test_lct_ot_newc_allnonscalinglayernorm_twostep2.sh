export MASTER_PORT=10124

epoch=1400
ema="ema_0.99993"

for ts in 460
do
    note=stochastic_twostep_${ts}

    CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
            --ckpt ./results/latent_celeb256/large_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_newc_allnonscalinglayernorm/checkpoints/000${epoch}.pt \
            --seed 42 \
            --dataset latent_celeb256 \
            --image-size 32 \
            --num-in-channels 4 \
            --num-classes 0 \
            --steps 641 \
            --batch-size $((256*1)) \
            --num-channels 128 \
            --num-head-channels 64 \
            --num-res-blocks 4 \
            --resblock-updown \
            --model-type dhariwal_unet \
            --channel-mult 1,2,3,4 \
            --attention-resolutions 16,8 \
            --sampler multistep \
            --ts 0,${ts},640 \
            --normalize-matrix celeb256_stat.npy \
            --real-img-dir ~/datasets/celeba_256_jpg/ \
            --compute-fid \
            --ema $ema \
            --last-norm-type non-scaling-layer-norm \
            --block-norm-type non-scaling-layer-norm \
            --note $note

        python eval_toolbox/calc_metrics.py \
            --metrics pr50k3_full \
            --data ~/datasets/celeba_256_jpg \
            --mirror 1 \
            --gen_data generated_samples/latent_celeb256/explarge_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_newc_allnonscalinglayernorm_ep000${epoch}_${ema}_$note \
            --img_resolution 256
done

python ~/envs/slack_workflow/running_finished.py
