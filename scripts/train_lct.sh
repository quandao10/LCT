export MASTER_PORT=10122

CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm_latent.py \
        --exp dhariwal_unet_ict_large_batchsize_lr_decay_4k_epoch_normalize_adaptive \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
        --dataset latent_celeb256 \
        --results-dir /research/cbim/medical/qd66/lct_exp/ \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 0 \
        --weight-schedule ict \
        --loss-norm adaptive \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 10 \
        --end-scales 1280 \
        --noise-sampler ict \
        --global-batch-size $((384*1)) \
        --epochs $((4000*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 2 \
        --resblock-updown \
        --ict \
        --max-grad-norm 2.0 \
        --model-type dhariwal_unet \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix celeb256_stat.npy
        # --l2-reweight \
        # --model-ckpt /research/cbim/medical/qd66/lct_exp/latent_celeb256/ict/checkpoints/0001000.pt \
