# # # export MASTER_PORT=10120

# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10112 --nproc_per_node=1 train_cm_latent.py \
#         --exp 700ep_L_relu_eps1e-4_unnormalized_linear_norm_pixel_norm_nogradnorm  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_celeb256 \
#         --results-dir /research/cbim/medical/qd66/lct_v2/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --weight-schedule ict \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((48*1)) \
#         --epochs $((700*1)) \
#         --lr 1e-4 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 100.0 \
#         --model-type DiT-L/2 \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix celeb256_stat.npy \
#         --use-diffloss \
#         --ot-hard \
#         --diff-lamb 5 \
#         --c-by-loss-std \
#         --ckpt-every 5 \
#         --save-content-every 5 \
#         --plot-every 5 \
#         --eps 1e-4 \
#         --linear-act relu \
#         --wo-norm \
#         --attn-type flash \
        # --model-ckpt /research/cbim/medical/qd66/lct_v2/latent_celeb256/celeb_dit_best_setting_700ep_L_relu_eps1e-4_unnormalized_linear_norm/checkpoints/0000595.pt
        # --resume

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10110 --nproc_per_node=2 train_cm_latent.py \
#         --exp lsun_dit_best_setting_700ep_L_relu_eps1e-4_unnormalized_linear_norm  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_church256_flip \
#         --results-dir /research/cbim/medical/qd66/lct_v2/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --weight-schedule ict \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((72*2)) \
#         --epochs $((350*1)) \
#         --lr 1e-4 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 100.0 \
#         --model-type DiT-L/2 \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix latent_church256_flip_stat.npy \
#         --use-diffloss \
#         --ot-hard \
#         --diff-lamb 5 \
#         --c-by-loss-std \
#         --ckpt-every 5 \
#         --save-content-every 2 \
#         --plot-every 5 \
#         --eps 1e-4 \
#         --linear-act relu \
#         --wo-norm \
#         --attn-type flash \
#         --model-ckpt /research/cbim/medical/qd66/lct_v2/latent_church256_flip/lsun_dit_best_setting_700ep_L_relu_eps1e-4_unnormalized_linear_norm/checkpoints/0000295.pt \
        # --final-conv \
        # --num-register 4 \
        # --model-ckpt /research/cbim/medical/qd66/lct_v2/latent_celeb256/celeb_dit_best_setting_700ep_B_relu_eps1e-5_unnormalized/checkpoints/0000600.pt \
        # --flash \
        # --use-scale-residual \
        
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10130 --nproc_per_node=8 train_cm_latent.py \
#         --exp imagenet_dit_best_setting_700ep_L_relu_eps1e-4_unnormalized_linear_norm  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_imagenet256 \
#         --results-dir /research/cbim/medical/qd66/lct_v2/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 1000 \
#         --weight-schedule ict \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((80*8)) \
#         --epochs $((350*1)) \
#         --lr 5e-5 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 40.0 \
#         --model-type DiT-L/2 \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix latent_imagenet256_stat.npy \
#         --use-diffloss \
#         --ot-hard \
#         --diff-lamb 5 \
#         --c-by-loss-std \
#         --ckpt-every 1 \
#         --save-content-every 1 \
#         --plot-every 1000 \
#         --eps 1e-4 \
#         --linear-act relu \
#         --wo-norm \
#         --attn-type flash \
#         --cfg-scale 1.0 \
#         --resume \


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10130 --nproc_per_node=8 train_cm_latent.py \
        --exp imagenet_rebutt  \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
        --dataset compress_latent_imagenet512 \
        --results-dir /research/cbim/medical/qd66/lct_v2/ \
        --image-size 16 \
        --num-in-channels 32 \
        --num-classes 1000 \
        --weight-schedule ict \
        --loss-norm cauchy \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 10 \
        --end-scales 640 \
        --noise-sampler ict \
        --global-batch-size $((512*8)) \
        --epochs $((1750*1)) \
        --lr 5e-5 \
        --num-sampling 8 \
        --num-channels 192 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100.0 \
        --model-type dhariwal_unet \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix compress_latent_imagenet512.npy \
        --use-diffloss \
        --ot-hard \
        --diff-lamb 5 \
        --c-by-loss-std \
        --ckpt-every 5 \
        --save-content-every 5 \
        --plot-every 1 \
        --eps 1e-4 \
        --linear-act relu \
        --wo-norm \
        --attn-type flash \
        --cfg-scale 1.0 \


# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10123 --nproc_per_node=1 train_cm_latent.py \
#         --exp celeb_dit_best_setting_700ep_B_relu_small_lr  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_celeb256 \
#         --results-dir ./results/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --weight-schedule ict \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((24*2)) \
#         --epochs $((700*1)) \
#         --lr 3e-5 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 100.0 \
#         --model-type DiT-B/2 \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix celeb256_stat.npy \
#         --use-diffloss \
#         --ot-hard \
#         --c-by-loss-std \
#         --linear-act relu \
        # --no-scale
#         # --resume


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10120 --nproc_per_node=8 train_cm_latent.py \
#         --exp lsun_baseline  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_church256_flip \
#         --results-dir ./results/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --weight-schedule ict \
#         --loss-norm huber \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((256*8)) \
#         --epochs $((1400*1)) \
#         --lr 1e-4 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 100.0 \
#         --model-type dhariwal_unet \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix latent_church256_flip_stat.npy \
#         --ckpt-every 25

# python ~/envs/slack_workflow/running_finished.py        


# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10120 --nproc_per_node=2 train_cm_latent.py \
#         --exp ffhq_best_setting  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_ffhq256 \
#         --results-dir ./results/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 0 \
#         --weight-schedule ict \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((192*8)) \
#         --epochs $((1750*1)) \
#         --lr 1e-4 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 100.0 \
#         --model-type dhariwal_unet \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix latent_ffhq256_stat.npy \
#         --use-diffloss \
#         --ot-hard \
#         --c-by-loss-std \