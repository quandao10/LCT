export MASTER_PORT=10125

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm_latent.py \
        --exp large_dhariwal_unet_cauchy_no_grad_norm_bs128_origincm_latentlpips \
        --datadir ./dataset/ \
        --dataset latent_celeb256 \
        --results-dir ./results/ \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 0 \
        --weight-schedule uniform \
        --loss-norm latent_lpips \
        --target-ema-mode adaptive \
        --start-ema 0.9 \
        --scale-mode progressive \
        --start-scales 2 \
        --end-scales 150 \
        --noise-sampler uniform \
        --global-batch-size $((128*1)) \
        --epochs $((1400*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --max-grad-norm 100.0 \
        --model-type dhariwal_unet \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --last-norm-type group-norm \
        --block-norm-type group-norm \
        --model-ckpt results/latent_celeb256/large_dhariwal_unet_cauchy_no_grad_norm_bs128_origincm_latentlpips/checkpoints/0000675.pt \
        # --resume \

python ~/envs/slack_workflow/running_finished.py        
