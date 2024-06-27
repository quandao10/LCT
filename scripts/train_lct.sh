export MASTER_PORT=10121

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm_latent.py \
        --exp ict \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
        --dataset latent_celeb256 \
        --results-dir /research/cbim/medical/qd66/lct_exp/ \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes -1 \
        --weight-schedule ict \
        --loss-norm huber \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 10 \
        --end-scales 1280 \
        --noise-sampler ict \
        --global-batch-size $((64*1)) \
        --epochs $((2000*1)) \
        --lr 0.00001 \
        --num-sampling 8 \
        --num-channels 192 \
        --num-head-channels 64 \
        --num-res-blocks 2 \
        --resblock-updown \
        --ict \
        --max-grad-norm 2.0 \
        --model-ckpt /research/cbim/medical/qd66/lct_exp/latent_celeb256/ict/checkpoints/0001000.pt \
