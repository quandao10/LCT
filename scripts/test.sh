export MASTER_PORT=10121

LOSS_NORM=l2
START_SCALES=40
END_SCALES=40
P_MEAN=-1.1
P_STD=2.0
LR=0.00002

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm_latent.py \
        --exp CT1/loss-norm=$LOSS_NORM/start-scales=$START_SCALES-end-scales=$END_SCALES/p-mean=$P_MEAN-p-std=$P_STD/lr=$LR \
        --datadir features/celeba256.npy \
        --dataset latent_celeb256 \
        --num-classes -1 \
        --weight-schedule uniform \
        --loss-norm $LOSS_NORM \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales $START_SCALES \
        --end-scales $END_SCALES \
        --global-batch-size 96 \
        --epochs 800 \
        --lr $LR \
        --num-sampling 8 \
        --image-size 32 \
        --num-in-channels 4 \
        --num-channels 192 \
        --num-head-channels 64 \
        --num-res-blocks 2 \
        --resblock-updown \
        --p-mean $P_MEAN \
        --p-std $P_STD \
        
python ~/envs/slack_workflow/running_finished.py
