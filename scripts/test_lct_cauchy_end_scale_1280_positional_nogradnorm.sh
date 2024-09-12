export MASTER_PORT=10124

for epoch in 1175 1375 1400 1600
do
        CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
                --ckpt results/latent_celeb256/dhariwal_unet_ict_large_batchsize_lr_decay_1k6_epoch_normalize_0.5_1280_nogradnorm_cauchy_positional/checkpoints/000${epoch}.pt \
                --seed 42 \
                --dataset latent_celeb256 \
                --image-size 32 \
                --num-in-channels 4 \
                --num-classes 0 \
                --steps 160 \
                --batch-size $((128*1)) \
                --num-channels 128 \
                --num-head-channels 64 \
                --num-res-blocks 2 \
                --resblock-updown \
                --model-type dhariwal_unet \
                --channel-mult 1,2,3,4 \
                --attention-resolutions 16,8 \
                --sampler onestep \
                --ts 0,9,19,39,79,160 \
                --normalize-matrix celeb256_stat.npy \
                --ema \
                --compute-fid \
                --time-emb positional
done

python ~/envs/slack_workflow/running_finished.py
