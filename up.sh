sshpass -p 'User@2023!@1' rsync -aPvr \
    --exclude .git \
    --exclude "*.pt" \
    --exclude "*.png" \
    --exclude "results" \
    --exclude "slurm" \
    ./ \
    superpod:/lustre/scratch/client/movian/research/users/anhnd72/projects/LCT/