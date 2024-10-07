#!/bin/bash -e
#SBATCH --job-name=sbatch-1                                                                 # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/thienlt3/khanhdn10/lct/logs/mbpp%A.out     # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/thienlt3/khanhdn10/lct/logs/mbpp%A.err      # create a error file
#SBATCH --partition=research                                                                # choose partition
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=40GB
#SBATCH --nodes=1
#SBATCH --nodelist=sdc2-hpc-dgx-a100-017
#SBATCH --ntasks=1
#SBATCH --mail-type=begin                                                                   # send email when job begins
#SBATCH --mail-type=end                                                                     # send email when job ends
#SBATCH --mail-type=fail                                                                    # send email when job fails
#SBATCH --mail-user=v.khanhdn10@vinai.io

# conda env
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"
conda activate /lustre/scratch/client/vinai/users/thienlt3/envs/khanhdn10/lct

export MASTER_PORT=10125

CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=2 train_cm_latent.py \
        --exp large_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_newc_allnonscalinglayernorm \
        --datadir ./dataset/ \
        --dataset latent_ffhq256 \
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
        --global-batch-size $((128*2)) \
        --epochs $((1400*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100.0 \
        --model-type dhariwal_unet \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix ffhq256_stat.npy \
        --use-diffloss \
        --ot-hard \
        --c-by-loss-std \
        --last-norm-type non-scaling-layer-norm \
        --block-norm-type non-scaling-layer-norm \
        # --resume \

python ~/envs/slack_workflow/running_finished.py        
