export MASTER_PORT=10120

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm_latent.py \
        --exp edm2_s_cauchy_no_grad_norm_bs128 \
        --datadir ./dataset/ \
        --dataset latent_celeb256 \
        --results-dir ./results/ \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 0 \
        --weight-schedule ict \
        --loss-norm cauchy \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 10 \
        --end-scales 640 \
        --noise-sampler ict \
        --global-batch-size $((128*1)) \
        --epochs $((1400*1)) \
        --lr 1e-2 \
        --num-sampling 8 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100.0 \
        --model-type EDM2-XS \
        --normalize-matrix celeb256_stat.npy \
        # --resume
        # --model-ckpt /research/cbim/medical/qd66/lct_exp/latent_celeb256/large_dhariwal_unet_cauchy_no_grad_norm_diff_0.75/checkpoints/0000975.pt \
        # --resume \

python ~/envs/slack_workflow/running_finished.py        
