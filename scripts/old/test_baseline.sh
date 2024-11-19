export MASTER_PORT=10123

# ema="ema"

# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 test_cm_latent_ddp.py \
#         --ckpt ./results/latent_celeb256/dhariwal_unet_ict_large_batchsize_lr_decay_1k6_epoch_normalize_0.5_1280_nogradnorm_huber_positional/checkpoints/content.pth \
#         --seed 42 \
#         --dataset latent_celeb256 \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --steps 1281 \
#         --batch-size $((256*1)) \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 2 \
#         --resblock-updown \
#         --model-type dhariwal_unet \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --sampler multistep \
#         --ts 0,640,1280 \
#         --normalize-matrix celeb256_stat.npy \
#         --real-img-dir ~/datasets/celeba_256_jpg/ \
#         --compute-fid \
#         --ema $ema \

python eval_toolbox/calc_metrics.py \
    --metrics pr50k3_full \
    --data ~/datasets/celeba_256_jpg \
    --mirror 1 \
    --gen_data generated_samples/latent_celeb256/expdhariwal_unet_ict_large_batchsize_lr_decay_1k6_epoch_normalize_0.5_1280_nogradnorm_huber_positional_epcontent._ema \
    --img_resolution 256

python ~/envs/slack_workflow/running_finished.py
