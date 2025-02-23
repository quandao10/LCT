export MASTER_PORT=10130

# for epoch in 625 650 675 700
# do
CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_cond.py \
        --ckpt /research/cbim/medical/qd66/lct_v2/subset_imagenet_256/im_700ep_B_relu_eps1e-4_repa_register_4/checkpoints/0000225.pt \
        --seed 42 \
        --dataset latent_imagenet256 \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 25 \
        --steps 641 \
        --batch-size $((16)) \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --model-type DiT-B/2 \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --sampler onestep \
        --ts 0,420,640 \
        --normalize-matrix statistic/stats_25.npy \
        --real-img-dir ../real_samples/celeba_256/ \
        --compute-fid \
        --ema \
        --linear-act relu \
        --num-register 4 \
        --cfg-scale 5 \
# done