export MASTER_PORT=10121

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm_latent.py \
        --exp large_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_newc_resumefrom41nfe \
        --datadir ./dataset/ \
        --dataset latent_celeb256 \
        --results-dir ./results/ \
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
        --global-batch-size $((128*1)) \
        --epochs $((1400*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100.0 \
        --model-type dhariwal_unet \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix celeb256_stat.npy \
        --use-diffloss \
        --ot-hard \
        --custom-constant-c 0.00614442 \
        --model-ckpt results/latent_celeb256/large_dhariwal_unet_cauchy_no_grad_norm_diff_0.75_newdiff_fix_5_bs128_othard_newc_resumefrom41nfe/checkpoints/0000575.pt \
        # --resume \

python ~/envs/slack_workflow/running_finished.py        
