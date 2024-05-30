export MASTER_PORT=10120

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm.py \
        --exp exp1 \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
        --dataset cifar10 \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes -1 \
        --weight-schedule uniform \
        --loss-norm huber \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 2 \
        --end-scales 200 \
        --global-batch-size 96 \
        --epochs 2000 \
        --lr 0.00002 \
        --num-channels 192 \
        --num-head-channels 64 \
        --num-res-blocks 2 \
        --resblock-updown \
        
