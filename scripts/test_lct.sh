export MASTER_PORT=10121

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 test_cm_latent_ddp.py \
        --ckpt /research/cbim/medical/qd66/lct_exp/latent_celeb256/large_dhariwal_unet_cauchy_no_grad_norm/checkpoints/0001000.pt \
        --seed 42 \
        --dataset latent_celeb256 \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 0 \
        --steps 161 \
        --batch-size $((192*1)) \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --model-type dhariwal_unet \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --sampler onestep \
        --ts 0,9,19,39,79,159 \
        --normalize-matrix celeb256_stat.npy \
        --ema \
        --real-img-dir /research/cbim/vast/qd66/workspace/real_samples/celeba_256 \
        --compute-fid \
        # --test-interval \
