export MASTER_PORT=10128

for epoch in 600 700 800 
do
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=8 test_diff_latent.py \
                --ckpt ./results_diff/latent_church256_flip/lsun_diff/checkpoints/0000${epoch}.pt \
                --seed 42 \
                --dataset latent_church256_diff \
                --image-size 32 \
                --num-in-channels 4 \
                --num-classes 0 \
                --steps 641 \
                --batch-size $((256*1)) \
                --num-channels 128 \
                --num-head-channels 64 \
                --num-res-blocks 4 \
                --resblock-updown \
                --model-type dhariwal_unet \
                --channel-mult 1,2,3,4 \
                --attention-resolutions 16,8 \
                --normalize-matrix latent_church256_flip_stat.npy \
                --real-img-dir ../real_samples/lsun/ \
                --compute-fid \
                --ema \
                # --test-interval \
done

python ~/envs/slack_workflow/running_finished.py
