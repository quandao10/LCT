#!/bin/bash -e
#SBATCH --job-name=sbatch-0                                                                 # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/thienlt3/khanhdn10/lct/logs/mbpp%A.out  # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/thienlt3/khanhdn10/lct/logs/mbpp%A.err   # create a error file
#SBATCH --partition=research                                                                # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-gpu=40GB
#SBATCH --nodes=1
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

# export MASTER_PORT=10123

# # for epoch in 0175 0375 0575 0775 0975 1175 1375
# for epoch in 1400
# do
#     for ema in "ema_0.99993"
#     do
#         CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
#                 --ckpt ./results/latent_celeb256/large_dhariwal_unet_cauchy_no_grad_norm_bs128_origincm_l2_new/checkpoints/000${epoch}.pt \
#                 --seed 42 \
#                 --dataset latent_celeb256 \
#                 --image-size 32 \
#                 --num-in-channels 4 \
#                 --num-classes 0 \
#                 --steps 161 \
#                 --batch-size $((256*1)) \
#                 --num-channels 128 \
#                 --num-head-channels 64 \
#                 --num-res-blocks 4 \
#                 --resblock-updown \
#                 --model-type dhariwal_unet \
#                 --channel-mult 1,2,3,4 \
#                 --attention-resolutions 16,8 \
#                 --sampler onestep \
#                 --ts 0,9,19,39,79,159 \
#                 --normalize-matrix celeb256_stat.npy \
#                 --real-img-dir ~/datasets/celeba_256_jpg/ \
#                 --compute-fid \
#                 --ema $ema \
#                 --last-norm-type non-scaling-layer-norm \
#                 --block-norm-type non-scaling-layer-norm \

#             python eval_toolbox/calc_metrics.py \
#                 --metrics pr50k3_full \
#                 --data ~/datasets/celeba_256_jpg \
#                 --mirror 1 \
#                 --gen_data generated_samples/latent_celeb256/explarge_dhariwal_unet_cauchy_no_grad_norm_bs128_origincm_l2_new_ep000${epoch}_${ema} \
#                 --img_resolution 256
#     done
# done


export MASTER_PORT=10124

# for epoch in 0175 0375 0575 0775 0975 1175 1375
for epoch in 1400
do
    for ema in "model"
    do
        CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
                --ckpt ./results/latent_celeb256/large_dhariwal_unet_cauchy_no_grad_norm_bs128_origincm_latentlpips_new/checkpoints/000${epoch}.pt \
                --seed 42 \
                --dataset latent_celeb256 \
                --image-size 32 \
                --num-in-channels 4 \
                --num-classes 0 \
                --steps 161 \
                --batch-size $((256*1)) \
                --num-channels 128 \
                --num-head-channels 64 \
                --num-res-blocks 4 \
                --resblock-updown \
                --model-type dhariwal_unet \
                --channel-mult 1,2,3,4 \
                --attention-resolutions 16,8 \
                --sampler onestep \
                --ts 0,9,19,39,79,159 \
                --normalize-matrix celeb256_stat.npy \
                --real-img-dir ~/datasets/celeba_256_jpg/ \
                --compute-fid \
                --ema $ema \
                --last-norm-type non-scaling-layer-norm \
                --block-norm-type non-scaling-layer-norm \

            # python eval_toolbox/calc_metrics.py \
            #     --metrics pr50k3_full \
            #     --data ~/datasets/celeba_256_jpg \
            #     --mirror 1 \
            #     --gen_data generated_samples/latent_celeb256/explarge_dhariwal_unet_cauchy_no_grad_norm_bs128_origincm_latentlpips_new_ep000${epoch}_${ema} \
            #     --img_resolution 256
    done
done

python ~/envs/slack_workflow/running_finished.py
