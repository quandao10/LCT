PREFIX=/home/khanhdn10/repo/lct

# torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10123 --nproc_per_node=1 train_cm_latent.py \
torchrun --nnodes=1 --nproc_per_node=1 train_cm_latent.py \
        --exp celeb_dit_best_setting_700ep_B_relu_small_lr  \
        --datadir $PREFIX/dataset/ \
        --dataset latent_celeb256 \
        --results-dir results/ \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 0 \
        --weight-schedule ict \
        --loss-norm cauchy \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 10 \
        --end-scales 640 \
        --noise-sampler ict \
        --global-batch-size $((24*2)) \
        --epochs $((700*1)) \
        --lr 3e-5 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100.0 \
        --model-type DiT-B/2 \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix celeb256_stat.npy \
        --use-diffloss \
        --ot-hard \
        --c-by-loss-std \
        --linear-act relu \
        --no-scale \
        --vae-type mit-han-lab/dc-ae-f32c32-in-1.0
        # --resume