#!/bin/sh
#SBATCH --job-name=one # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/haopt12/swiftbrush-code/slurms/slurm_%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/haopt12/swiftbrush-code/slurms/slurm_%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16 # 80
#SBATCH --mem-per-gpu=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.haopt12@vinai.io

set -x
set -e

export MASTER_PORT=10112
export WORLD_SIZE=1

export SLURM_JOB_NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')
export SLURM_NODELIST=$SLURM_JOB_NODELIST
master_address=$(echo $SLURM_JOB_NODELIST | cut -d' ' -f1)
export MASTER_ADDRESS=$master_address

echo MASTER_ADDRESS=${MASTER_ADDRESS}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${WORLD_SIZE}
echo "NODELIST="${SLURM_NODELIST}

# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
#--rdzv_endpoint 0.0.0.0:8000
export OMP_NUM_THREADS=24


CUDA_VISIBLE_DEVICES=3,4 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:8005 --nproc_per_node=2 vim/train.py \
                                                                                            --exp L_2_linear_block_cpe_dilated_style \
                                                                                            --model DiM-L/2_Jamba \
                                                                                            --datadir ../dataset/celeba-lmdb/ \
                                                                                            --dataset celeba_256 \
                                                                                            --global-batch-size 48 \
                                                                                            --lr 1e-4 \
                                                                                            --epochs 500 \
                                                                                            --learn-sigma \
                                                                                            --pe-type ape \
                                                                                            --block-type linear \
                                                                                            --no-lr-decay \
                                                                                            --eval-every 50 \
                                                                                            --eval-refdir ./real_samples/celeba_256 \
                                                                                            --eval-nsamples 2000 \
                                                                                            --eval-bs 20 \
                                                                                            --eval-start-epoch 300 \
                                                                                            --ckpt-every 25 \
                                                                                            # --use_dilated \
                                                                                            # --resume ./results/L_2_linear_block_cpe_dilated-DiM-L-2/checkpoints/0000250.pt \
                                                                                            # --using_dct \ # --use_dilated \ --use_wavelet \