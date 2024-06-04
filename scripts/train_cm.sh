export MASTER_PORT=10122

# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm.py \
#         --exp exp1 \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset cifar10 \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes -1 \
#         --weight-schedule uniform \
#         --loss-norm huber \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 2 \
#         --end-scales 200 \
#         --global-batch-size 96 \
#         --epochs 2000 \
#         --lr 0.00002 \
#         --num-channels 192 \
#         --num-head-channels 64 \
#         --num-res-blocks 2 \
#         --resblock-updown \

### improved CT

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm.py \
        --exp icm_fix_sampling_dist_lr1e-4 \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
        --dataset cifar10 \
        --image-size 32 \
        --num-in-channels 3 \
        --num-classes -1 \
        --weight-schedule ict \
        --loss-norm huber \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode fixed \
        --start-scales 10 \
        --end-scales 1280 \
        --global-batch-size 196 \
        --epochs 1800 \
        --lr 0.0001 \
        --num-channels 192 \
        --num-head-channels 64 \
        --num-res-blocks 2 \
        --resblock-updown \
        --dropout 0.3 \

        
