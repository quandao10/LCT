sshpass -p 'User@2023!@1' rsync -aPvr \
--exclude "local_datasets" \
--exclude "checkpoints" \
--exclude "data" \
--exclude "*.pth" \
--exclude "*.pt" \
--exclude .git \
superpod:/lustre/scratch/client/movian/research/users/anhnd72/projects/LCT/results/ \
results/