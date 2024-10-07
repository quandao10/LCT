# # # export MASTER_PORT=10120

# export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10120 --nproc_per_node=1 train_cm_latent.py \
#         --exp celeb_dit_best_setting  \
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
#         --global-batch-size $((32*2)) \
#         --epochs $((1400*1)) \
#         --lr 1e-4 \
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
#         --resume


# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10122 --nproc_per_node=1 train_cm_latent.py \
#         --exp celeb_dit_best_setting_non_scale  \
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
#         --global-batch-size $((32*2)) \
#         --epochs $((1400*1)) \
#         --lr 1e-4 \
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
#         --resume


# CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10121 --nproc_per_node=1 train_cm_latent.py \
#         --exp celeb_dit_huber  \
#         --datadir /research/cbim/vast/qd66/workspace/dataset/ \
#         --dataset latent_celeb256 \
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
#         --global-batch-size $((32*2)) \
#         --epochs $((1400*1)) \
#         --lr 1e-4 \
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
#         # --c-by-loss-std \
        # --use-diffloss \
        # --ot-hard \


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