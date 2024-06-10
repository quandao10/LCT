export MASTER_PORT=10120

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

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 train_cm.py \
        --exp icm_ict_weighting_dist_bigbatchsize \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
        --dataset cifar10 \
        --results-dir /research/cbim/medical/qd66/lct_exp/ \
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
        --global-batch-size $((200*4)) \
        --epochs $((1500*4)) \
        --lr 0.0001 \
        --num-channels 192 \
        --num-head-channels 64 \
        --num-res-blocks 2 \
        --resblock-updown \
        --dropout 0.3 \

        
