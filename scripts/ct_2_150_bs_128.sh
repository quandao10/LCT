export MASTER_PORT=10123

LOSS_NORM=l2
START_SCALES=2
END_SCALES=150
P_MEAN=-1.1
P_STD=2.0
LR=0.0004
BATCH_SIZE=1024

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_ct.py \
        --exp CT1/loss-norm=$LOSS_NORM/start-scales=$START_SCALES-end-scales=$END_SCALES/p-mean=$P_MEAN-p-std=$P_STD/batch-size=$BATCH_SIZE/lr=$LR \
        --datadir dataset/ \
        --dataset cifar10 \
        --num-classes -1 \
        --log-every 10 \
        --weight-schedule uniform \
        --loss-norm $LOSS_NORM \
        --target-ema-mode adaptive \
        --start-ema 0.9 \
        --scale-mode progressive \
        --start-scales $START_SCALES \
        --end-scales $END_SCALES \
        --global-batch-size 128 \
        --desired-batch-size $BATCH_SIZE \
        --epochs 8334 \
        --lr $LR \
        --image-size 32 \
        --num-in-channels 3 \
        --num-channels 192 \
        --num-head-channels 64 \
        --num-res-blocks 2 \
        --resblock-updown \
        --dropout 0.3 \
        --p-mean $P_MEAN \
        --p-std $P_STD \
        
python ~/envs/slack_workflow/running_finished.py
