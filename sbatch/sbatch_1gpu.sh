#!/bin/bash -lex
# SLURM SUBMIT SCRIPT
#SBATCH --job-name=LCT
#SBATCH --output=/lustre/scratch/client/movian/research/users/anhnd72/projects/LCT/sbatch/slurm_anhnd72.log
#SBATCH --error=/lustre/scratch/client/movian/research/users/anhnd72/projects/LCT/sbatch/slurm_anhnd72.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-gpu=100GB
#SBATCH --partition=research
#SBATCH --mail-type=all
#SBATCH --mail-user=v.anhnd72@vinai.io

# (optional) debugging flags
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1
# export NCCL_SOCKET_IFNAME=bond0

   
module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"

conda activate /lustre/scratch/client/movian/research/users/anhnd72/envs/torch22_cloned4_chitb
cd /lustre/scratch/client/movian/research/users/anhnd72/projects/LCT/

# export PYTHONUSERBASE=intentionally-disabled
# python -c "import site; print(site.USER_SITE)" intentionally-disabled/lib/python3.7/site-packages


echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=31220

echo "MASTER_ADDR="$MASTER_ADDR
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$MASTER_ADDR" hostname --ip-address)
echo "head_node_ip="$head_node_ip

bash scripts/train_lct_dit_repa.sh 1