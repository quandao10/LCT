#!/bin/bash
#SBATCH --mail-user=htp26@cornell.edu  # Email
#SBATCH --mail-type=END               # Request status by email
#SBATCH -J lct             # Job name
#SBATCH -o watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=32000                   # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a5000:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export MASTER_PORT=10120

eval "$(conda shell.bash hook)"
conda activate lct

BATCH_SIZE=128
NUM_GPUS=1
EXP=celeb_ditb_flashattn_700ep_relu_v1

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=$NUM_GPUS train_cm_latent.py \
        --exp ${EXP} \
        --datadir /share/kuleshov/datasets/latent_celeba_256/ \
        --dataset latent_celeb256 \
        --results-dir /share/kuleshov/htp26/lct/results/ \
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
        --global-batch-size $((BATCH_SIZE * NUM_GPUS)) \
        --epochs $((700*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100.0 \
        --model-type DiT-B/2 \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix celeb256_stat.npy \
        --use-diffloss \
        --ot-hard \
        --c-by-loss-std \
        --linear-act relu \
        --block-type DiTBlockFlashAttn \
        --sampler euler \
        --steps 10 \
        # --resume \
        # --wo-norm \
        # --use-scale-residual \
        



# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10123 --nproc_per_node=1 train_cm_latent.py \
#         --exp celeb_dit_best_setting_700ep_B_relu_small_lr  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_celeb256 \
#         --results-dir ./results/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --weight-schedule ict \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((24*2)) \
#         --epochs $((700*1)) \
#         --lr 3e-5 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 100.0 \
#         --model-type DiT-B/2 \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix celeb256_stat.npy \
#         --use-diffloss \
#         --ot-hard \
#         --c-by-loss-std \
#         --linear-act relu \
        # --no-scale
#         # --resume


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10120 --nproc_per_node=8 train_cm_latent.py \
#         --exp lsun_baseline  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_church256_flip \
#         --results-dir ./results/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --weight-schedule ict \
#         --loss-norm huber \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((256*8)) \
#         --epochs $((1400*1)) \
#         --lr 1e-4 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 100.0 \
#         --model-type dhariwal_unet \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix latent_church256_flip_stat.npy \
#         --ckpt-every 25

# python ~/envs/slack_workflow/running_finished.py        


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10120 --nproc_per_node=2 train_cm_latent.py \
#         --exp ffhq_best_setting  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_ffhq256 \
#         --results-dir ./results/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --weight-schedule ict \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((192*8)) \
#         --epochs $((1750*1)) \
#         --lr 1e-4 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 100.0 \
#         --model-type dhariwal_unet \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix latent_ffhq256_stat.npy \
#         --use-diffloss \
#         --ot-hard \
#         --c-by-loss-std \