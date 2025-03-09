# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10200 --nproc_per_node=2 train_cm_latent_repa.py \
#         --exp im_700ep_XL_relu_eps1e-4_repa_register_2 \
#         --datadir /common/users/qd66/repa/latent_imagenet256  \
#         --dataset subset_imagenet_256 \
#         --results-dir /research/cbim/medical/qd66/lct_v2/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 25 \
#         --weight-schedule ict \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 10 \
#         --end-scales 640 \
#         --noise-sampler ict \
#         --global-batch-size $((48)) \
#         --epochs $((700*1)) \
#         --lr 1e-4 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm 100 \
#         --model-type DiT-XL/2 \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix statistic/stats_25.npy \
#         --use-diffloss \
#         --ot-hard \
#         --c-by-loss-std \
#         --linear-act relu \
#         --attn-type normal \
#         --projector-dim 2048 \
#         --repa-lamb 0.5 \
#         --repa-enc-info 8:dinov2-vit-b \
#         --repa-relu-margin 0.4 \
#         --repa-timesteps full \
#         --denoising-task-rate 0.5 \
#         --repa-mapper repa \
#         --mar-mapper-num-res-blocks 0 \
#         --use-repa \
#         --num-register 2 \
#         --resume



CUDA_VISIBLE_DEVICES=4,5 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10228 --nproc_per_node=2 train_cm_latent_repa.py \
        --exp im_700ep_lightningDiT_repa_register_0_B_wavelet_gate_relu \
        --datadir /common/users/qd66/repa/latent_imagenet256  \
        --dataset subset_imagenet_256 \
        --results-dir /research/cbim/medical/qd66/lct_v2/ \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 25 \
        --weight-schedule ict \
        --loss-norm cauchy \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 10 \
        --end-scales 640 \
        --noise-sampler ict \
        --global-batch-size $((48)) \
        --epochs $((700*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100 \
        --model-type DiT-B/2 \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix statistic/stats_25.npy \
        --use-diffloss \
        --ot-hard \
        --c-by-loss-std \
        --linear-act gate_wavelet_relu \
        --norm-type rms \
        --projector-dim 2048 \
        --repa-lamb 0.5 \
        --repa-enc-info 4:dinov2-vit-b \
        --repa-relu-margin 0.4 \
        --repa-timesteps full \
        --denoising-task-rate 0.5 \
        --repa-mapper repa \
        --mar-mapper-num-res-blocks 0 \
        --num-register 0 \
        --use-rope \
        --use-repa \
        --cond-type adain \
        # --use-freq-cond \
        # --resume
