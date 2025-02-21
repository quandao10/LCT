REPALAMB=0.1 # >0 --> use REPA
REPA_ENC_INFO="4:dinov2-vit-b" # it means: use dinov2-vit-b at 4-th layer
PROJ_DIM=2048 # 2048
MAPPER="repa" # repa, mar
MAR_MAPPER_NUM_RES_BLOCKS=0 # 1 2
REPA_TIMESTEPs="generation" # (full), denoising, generation
REPA_RELU_MARGIN=0.4 # clamping
DENOISING_TASK_RATE=0.5 # e.g., 0.25 --> definition of denoising task is the first 25% of the timesteps



CUDA_VISIBLE_DEVICES=5 torchrun --nnodes=1 --nproc_per_node=1 train_cm_latent_repa.py \
        --exp 700ep_B_relu_eps1e-4_repa \
        --datadir /common/users/qd66/repa/latent_celeb256  \
        --dataset latent_celeb256 \
        --results-dir /research/cbim/medical/qd66/lct_v2/ \
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
        --global-batch-size $((48)) \
        --epochs $((700*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 0 \
        --model-type DiT-B/2 \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix statistic/celeb256_stat.npy \
        --use-diffloss \
        --ot-hard \
        --c-by-loss-std \
        --linear-act relu \
        --attn-type normal \
        --projector-dim 2048 \
        --repa-lamb 0.1 \
        --repa-enc-info 4:dinov2-vit-b \
        --repa-relu-margin 0.4 \
        --repa-timesteps generation \
        --denoising-task-rate 0.5 \
        --repa-mapper repa \
        --mar-mapper-num-res-blocks 0 \
        --use-repa \
        
