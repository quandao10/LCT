export MASTER_PORT=10123

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
        --ckpt /research/cbim/medical/qd66/lct_exp/latent_celeb256/dhariwal_unet_ict/checkpoints/0001000.pt \
        --dataset latent_celeb256 \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 0 \
        --steps 1280 \
        --batch-size $((8*1)) \
        --num-channels 192 \
        --num-head-channels 64 \
        --num-res-blocks 2 \
        --resblock-updown \
        --model-type dhariwal_unet \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 32,16,8 \
        --sampler multistep \
        --ts 0,160,320,640,960,1279 \
        --ema \
