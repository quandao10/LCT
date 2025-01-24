export MASTER_PORT=10128

for epoch in 2000
do
        CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
                --ckpt /research/cbim/medical/qd66/lct_v2/compress_latent_imagenet512/imagenet_rebutt_2/checkpoints/000${epoch}.pt \
                --seed 42 \
                --dataset compress_latent_imagenet512 \
                --image-size 16 \
                --num-in-channels 32 \
                --num-classes 1000 \
                --steps 1281 \
                --batch-size $((8*1)) \
                --num-channels 128 \
                --num-head-channels 64 \
                --num-res-blocks 3 \
                --resblock-updown \
                --model-type dhariwal_unet \
                --channel-mult 1,2,3,4 \
                --attention-resolutions 16,8 \
                --sampler onestep \
                --ts 0,641,1280 \
                --normalize-matrix compress_latent_imagenet512.npy \
                --real-img-dir ~/datasets/celeba_256_jpg/ \
                --ema \
                # --compute-fid \
                # --test-interval \
done

python ~/envs/slack_workflow/running_finished.py
