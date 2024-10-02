export MASTER_PORT=10129

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

### improved CT setting
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm_.py \
        --exp ict_no_dp \
        --datadir ./dataset/ \
        --dataset cifar10 \
        --results-dir ./results/ \
        --image-size 32 \
        --num-in-channels 3 \
        --num-classes -1 \
        --weight-schedule ict \
        --loss-norm huber \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 10 \
        --noise-sampler ict \
        --end-scales 1280 \
        --global-batch-size $((256*1)) \
        --epochs $((2000*1)) \
        --lr 0.00002 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 2 \
        --resblock-updown \
        --ict \
        --model-ckpt results/cifar10/ict_no_dp/checkpoints/0000025.pt \


### rerun consistency setting
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm.py \
#         --exp ct \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset cifar10 \
#         --results-dir /research/cbim/medical/qd66/lct_exp/ \
#         --image-size 32 \
#         --num-in-channels 3 \
#         --num-classes -1 \
#         --weight-schedule uniform \
#         --loss-norm huber \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 2 \
#         --end-scales 200 \
#         --noise-sampler uniform \
#         --global-batch-size $((96*1)) \
#         --epochs $((2000*1)) \
#         --lr 0.00002 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 2 \
#         --resblock-updown \
#         --resume

### rerun consistency setting
# CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm.py \
#         --exp ct_ict_weight+sampler \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset cifar10 \
#         --results-dir /research/cbim/medical/qd66/lct_exp/ \
#         --image-size 32 \
#         --num-in-channels 3 \
#         --num-classes -1 \
#         --weight-schedule ict \
#         --loss-norm huber \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 2 \
#         --end-scales 200 \
#         --noise-sampler ict \
#         --global-batch-size $((96*1)) \
#         --epochs $((2000*1)) \
#         --lr 0.00002 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 2 \
#         --resblock-updown \
#         --resume

### rerun consistency setting
# CUDA_VISIBLE_DEVICES=3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm.py \
#         --exp ct_ict_weight+sampler_ict_2_200 \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset cifar10 \
#         --results-dir /research/cbim/medical/qd66/lct_exp/ \
#         --image-size 32 \
#         --num-in-channels 3 \
#         --num-classes -1 \
#         --weight-schedule ict \
#         --loss-norm huber \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 2 \
#         --end-scales 200 \
#         --noise-sampler ict \
#         --global-batch-size $((96*1)) \
#         --epochs $((2000*1)) \
#         --lr 0.00002 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 2 \
#         --resblock-updown \
#         --ict \
#         --resume
        
