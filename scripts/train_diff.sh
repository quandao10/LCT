CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10120 --nproc_per_node=2 train_diff_latent.py \
        --exp celeb_diff_cauchy  \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
        --dataset latent_celeb256 \
        --results-dir ./results_diff/ \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 0 \
        --global-batch-size $((128*2)) \
        --epochs $((800*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --model-type dhariwal_unet \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix celeb256_stat.npy \


# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10122 --nproc_per_node=6 train_diff_latent.py \
#         --exp lsun_diff  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_church256_flip \
#         --results-dir ./results_diff/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --global-batch-size $((256*6)) \
#         --epochs $((800*1)) \
#         --lr 1e-4 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --model-type dhariwal_unet \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix latent_church256_flip_stat.npy \