sshpass -p 'User@2023!@1' rsync -aPvr \
    --exclude .git \
    --exclude "*.pt" \
    --exclude "*.png" \
    --exclude "results" \
    --exclude "slurm" \
    ./ \
    superpod:/lustre/scratch/client/movian/research/users/anhnd72/projects/LCT/


# sshpass -p 'User@2023!@12' rsync -aPvr \
#     --exclude .git \
#     --exclude "*.pt" \
#     --exclude "*.png" \
#     --exclude "results" \
#     --exclude "slurm" \
#     /Users/ducanhnguyen/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth \
#     superpod:/home/anhnd72/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
