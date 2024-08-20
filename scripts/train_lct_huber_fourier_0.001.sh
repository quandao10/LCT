export MASTER_PORT=10120

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm_latent.py \
        --exp dhariwal_unet_ict_large_batchsize_lr_decay_4k_epoch_normalize_0.5_huber_fourier_0.001 \
        --datadir ./dataset/ \
        --dataset latent_celeb256 \
        --results-dir ./results/ \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 0 \
        --weight-schedule ict \
        --loss-norm huber \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 10 \
        --end-scales 1280 \
        --noise-sampler ict \
        --global-batch-size $((192*1)) \
        --epochs $((1000*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 2.0 \
        --model-type dhariwal_unet \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix celeb256_stat.npy \
        --time-emb fourier \
        --fourier-time-emb-scale 0.001 \
        # --use-diffloss \
        # --l2-reweight \
        # --model-ckpt /research/cbim/medical/qd66/lct_exp/latent_celeb256/ict/checkpoints/0001000.pt \

python ~/envs/slack_workflow/running_finished.py
