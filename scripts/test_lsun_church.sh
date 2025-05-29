export MASTER_PORT=10132

for epoch in 600 625 650 675 700
do
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 test_cm_latent_ddp.py \
                --ckpt /research/cbim/medical/qd66/lct_v2/latent_church256_flip/700ep_B_repa_prevmlp_8_512_rope_700ep/checkpoints/0000${epoch}.pt \
                --seed 42 \
                --dataset lsun_church \
                --image-size 32 \
                --num-in-channels 4 \
                --num-classes 0 \
                --steps 513 \
                --batch-size $((64*1)) \
                --num-channels 128 \
                --num-head-channels 64 \
                --num-res-blocks 4 \
                --resblock-updown \
                --model-type DiT-B/2 \
                --channel-mult 1,2,3,4 \
                --attention-resolutions 16,8 \
                --sampler onestep \
                --ts 0,256,512 \
                --normalize-matrix statistic/latent_church256_flip_stat.npy \
                --real-img-dir ../real_samples/lsun/ \
                --ema \
                --linear-act relu \
                --num-register 0 \
                --norm-type rms \
                --freq-type prev_mlp \
                --use-rope \
                --vae vae \
                 --compute-fid \

                
done