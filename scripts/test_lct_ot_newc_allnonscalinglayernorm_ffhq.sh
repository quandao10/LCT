export MASTER_PORT=10124

for epoch in 1400
do
    for ema in "ema_0.999" "ema" "ema_0.99994" "ema_0.99995" "ema_0.99997" "ema_0.9999432189950708"
    do
        CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
                --ckpt ./results/latent_ffhq256/large_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_newc_allnonscalinglayernorm/checkpoints/000${epoch}.pt \
                --seed 42 \
                --dataset latent_ffhq256 \
                --image-size 32 \
                --num-in-channels 4 \
                --num-classes 0 \
                --steps 161 \
                --batch-size $((256*1)) \
                --num-channels 128 \
                --num-head-channels 64 \
                --num-res-blocks 4 \
                --resblock-updown \
                --model-type dhariwal_unet \
                --channel-mult 1,2,3,4 \
                --attention-resolutions 16,8 \
                --sampler onestep \
                --ts 0,9,19,39,79,159 \
                --normalize-matrix ffhq256_stat.npy \
                --real-img-dir ~/datasets/ffhq_256_jpg/ \
                --compute-fid \
                --ema $ema \
                --last-norm-type non-scaling-layer-norm \
                --block-norm-type non-scaling-layer-norm \

            python eval_toolbox/calc_metrics.py \
                --metrics pr50k3_full \
                --data ~/datasets/ffhq_256_jpg \
                --mirror 1 \
                --gen_data generated_samples/latent_ffhq256/explarge_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_newc_allnonscalinglayernorm_ep000${epoch}_${ema} \
                --img_resolution 256
    done
done

for epoch in 1375 1350 1325
do
    for ema in "ema_0.99993" "ema_0.999" "ema" "ema_0.99994" "ema_0.99995" "ema_0.99997" "ema_0.9999432189950708"
    do
        CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
                --ckpt ./results/latent_ffhq256/large_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_newc_allnonscalinglayernorm/checkpoints/000${epoch}.pt \
                --seed 42 \
                --dataset latent_ffhq256 \
                --image-size 32 \
                --num-in-channels 4 \
                --num-classes 0 \
                --steps 161 \
                --batch-size $((256*1)) \
                --num-channels 128 \
                --num-head-channels 64 \
                --num-res-blocks 4 \
                --resblock-updown \
                --model-type dhariwal_unet \
                --channel-mult 1,2,3,4 \
                --attention-resolutions 16,8 \
                --sampler onestep \
                --ts 0,9,19,39,79,159 \
                --normalize-matrix ffhq256_stat.npy \
                --real-img-dir ~/datasets/ffhq_256_jpg/ \
                --compute-fid \
                --ema $ema \
                --last-norm-type non-scaling-layer-norm \
                --block-norm-type non-scaling-layer-norm \

            python eval_toolbox/calc_metrics.py \
                --metrics pr50k3_full \
                --data ~/datasets/ffhq_256_jpg \
                --mirror 1 \
                --gen_data generated_samples/latent_ffhq256/explarge_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_newc_allnonscalinglayernorm_ep000${epoch}_${ema} \
                --img_resolution 256
    done
done

python ~/envs/slack_workflow/running_finished.py
