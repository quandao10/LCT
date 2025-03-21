export MASTER_PORT=10132

for epoch in 600 625 650 675
do
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 test_cm_latent_ddp.py \
                --ckpt /research/cbim/medical/qd66/lct_v2/imagenet_256/im_700ep_lightningDiT_repa_register_0_B_premlp_notgate/checkpoints/0000${epoch}.pt \
                --seed 42 \
                --dataset imagenet_256 \
                --image-size 32 \
                --num-in-channels 4 \
                --num-classes 1000 \
                --steps 513 \
                --batch-size $((256*1)) \
                --num-channels 128 \
                --num-head-channels 64 \
                --num-res-blocks 4 \
                --resblock-updown \
                --model-type DiT-B/2 \
                --channel-mult 1,2,3,4 \
                --attention-resolutions 16,8 \
                --sampler onestep \
                --ts 0,256,512 \
                --normalize-matrix statistic/stats_25.npy \
                --real-img-dir ../real_samples/celeba_256/ \
                --compute-fid \
                --ema \
                --linear-act relu \
                --num-register 0 \
                --norm-type rms \
                --freq-type prev_mlp \
                --use-rope \
                --cfg-scale 1.5 \
                # --wo-norm \
                # --no-scale \
done