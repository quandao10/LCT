#!/bin/bash -e
#SBATCH --job-name=sbatch-0                                                                 # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/khanhdn10/repo/lct2/logs/mbpp%A.out     # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/khanhdn10/repo/lct2/logs/mbpp%A.err      # create a error file
#SBATCH --partition=research                                                                # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=40GB
#SBATCH --nodes=1
#SBATCH --nodelist=sdc2-hpc-dgx-a100-015
#SBATCH --ntasks=1
#SBATCH --mail-type=begin                                                                   # send email when job begins
#SBATCH --mail-type=end                                                                     # send email when job ends
#SBATCH --mail-type=fail                                                                    # send email when job fails
#SBATCH --mail-user=v.khanhdn10@vinai.io

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/lustre/scratch/client/vinai/users/khanhdn10/envs/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/lustre/scratch/client/vinai/users/khanhdn10/envs/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/lustre/scratch/client/vinai/users/khanhdn10/envs/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/lustre/scratch/client/vinai/users/khanhdn10/envs/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

eval "$(conda shell.bash hook)"
conda activate lct

bash scripts/train_lct_ot_gm.sh    
