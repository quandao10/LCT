# CUDA_VISIBLE_DEVICES=4,5 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10204 --nproc_per_node=2 train_cm_latent_repa.py \
#         --exp eq_DiT_ve_repa_register_0_B_premlp_noot_karras-0.8,1.5_diff_0.70_rho7_min-snr_20_nogate \
#         --datadir /common/users/qd66/repa/latent_imagenet256  \
#         --dataset subset_imagenet_256 \
#         --results-dir /research/cbim/medical/qd66/lct_v2_new/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 25 \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 8 \
#         --end-scales 512 \
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
#         --max-grad-norm -1 \
#         --model-type DiT-B/2 \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix statistic/stats_25.npy \
#         --use-diffloss \
#         --c-by-loss-std \
#         --linear-act relu \
#         --norm-type rms \
#         --projector-dim 2048 \
#         --repa-lamb 0.1 \
#         --repa-enc-info 8:dinov2-vit-b \
#         --repa-relu-margin 0.4 \
#         --repa-timesteps full \
#         --denoising-task-rate 0.5 \
#         --repa-mapper repa \
#         --mar-mapper-num-res-blocks 0 \
#         --num-register 0 \
#         --freq-type prev_mlp \
#         --vae vae \
#         --fwd ve \
#         --diff-lamb 5 \
#         --c-type edm \
#         --opt radam \
#         --use-rope \
#         --cond-mixing \
#         --p-mean -0.8 \
#         --p-std 1.5 \
#         --sigma-data 0.5 \
#         --use-repa \
#         --diff-rate 0.70 \
#         --tau 20 \
#         --weight-schedule min-snr \
#         --a 0.9 \
#         --b 0.9 \
#         --rho 7 \
#         --compile \
        # --resume
        # --use-karras-normalization \
        # --record-loss \
        
        



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10212 --nproc_per_node=8 train_cm_latent_repa.py \
        --exp va_DiT_L_v2 \
        --num-workers 8 \
        --datadir ~/workspace/dataset/repa/latent_imagenet256  \
        --dataset imagenet_256_va \
        --results-dir /research/cbim/medical/qd66/lct_v2_new/ \
        --image-size 16 \
        --num-in-channels 32 \
        --num-classes 1000 \
        --weight-schedule ict \
        --loss-norm cauchy \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 8 \
        --end-scales 512 \
        --noise-sampler ict \
        --global-batch-size $((2048)) \
        --epochs $((700*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100 \
        --model-type DiT-L/2 \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix statistic/latents_stats.pt \
        --use-diffloss \
        --c-by-loss-std \
        --linear-act relu \
        --norm-type rms \
        --projector-dim 2048 \
        --repa-lamb 0.1 \
        --repa-enc-info 4:dinov2-vit-b \
        --repa-relu-margin 0.4 \
        --repa-timesteps full \
        --denoising-task-rate 0.5 \
        --repa-mapper repa \
        --mar-mapper-num-res-blocks 0 \
        --num-register 0 \
        --freq-type prev_mlp \
        --use-rope \
        --cond-mixing \
        --vae va_vae \
        --fwd ve \
        --p-mean -0.8 \
        --p-std 1.5 \
        --sigma-data 0.5 \
        --diff-rate 0.70 \
        --diff-lamb 5 \
        --c-type edm \
        --opt radam \
        --compile \
        --ckpt-every 1 \
        --plot-every 1 \
        # --model-ckpt /research/cbim/medical/qd66/lct_v2_new/imagenet_256_va/va_DiT_ve_repa_register_0_XL_premlp_noot_karras-0.8,1.5_diff_0.70_rho7_ict_resume440/checkpoints/0000530.pt \
# #         # --resume
# #         # --use-repa
# #         # --resume



# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10212 --nproc_per_node=8 train_cm_latent_repa.py \
#         --exp resume525_mean-0.8_std1.5_diff0.70_lamb50 \
#         --num-workers 8 \
#         --datadir ~/workspace/dataset/repa/latent_imagenet256  \
#         --dataset imagenet_256_va \
#         --results-dir /research/cbim/medical/qd66/lct_v2_new/ \
#         --image-size 16 \
#         --num-in-channels 32 \
#         --num-classes 1000 \
#         --weight-schedule ict \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 8 \
#         --end-scales 512 \
#         --noise-sampler ict \
#         --global-batch-size $((2048)) \
#         --epochs $((700*1)) \
#         --lr 1e-4 \
#         --num-sampling 8 \
#         --num-channels 128 \
#         --num-head-channels 64 \
#         --num-res-blocks 4 \
#         --resblock-updown \
#         --ict \
#         --max-grad-norm -1 \
#         --model-type DiT-XL/2 \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix statistic/latents_stats.pt \
#         --use-diffloss \
#         --c-by-loss-std \
#         --linear-act relu \
#         --norm-type rms \
#         --projector-dim 2048 \
#         --repa-lamb 0.1 \
#         --repa-enc-info 4:dinov2-vit-b \
#         --repa-relu-margin 0.4 \
#         --repa-timesteps full \
#         --denoising-task-rate 0.5 \
#         --repa-mapper repa \
#         --mar-mapper-num-res-blocks 0 \
#         --num-register 0 \
#         --freq-type prev_mlp \
#         --use-rope \
#         --cond-mixing \
#         --vae va_vae \
#         --fwd ve \
#         --p-mean -0.8 \
#         --p-std 1.5 \
#         --sigma-data 0.5 \
#         --diff-rate 0.70 \
#         --diff-lamb 50 \
#         --c-type edm \
#         --opt radam \
#         --compile \
#         --ckpt-every 1 \
#         --plot-every 1 \
#         --model-ckpt /research/cbim/medical/qd66/lct_v2_new/imagenet_256_va/va_DiT_ve_repa_register_0_L_premlp_noot_karras-0.8,1.5_diff_0.70_rho7_ict_nograd_resume500/checkpoints/0000525.pt \



# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10234 --nproc_per_node=4 train_cm_latent_repa.py \
#         --exp DiT_ict_scale \
#         --num-workers 8 \
#         --datadir ~/workspace/dataset/repa/latent_imagenet256  \
#         --dataset subset_imagenet_256 \
#         --results-dir /research/cbim/medical/qd66/lct_v2_new/ \
#         --image-size 32 \
#         --num-in-channels 4 \
#         --num-classes 25 \
#         --weight-schedule ict_scale \
#         --loss-norm cauchy \
#         --target-ema-mode adaptive \
#         --start-ema 0.95 \
#         --scale-mode progressive \
#         --start-scales 8 \
#         --end-scales 512 \
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
#         --model-type DiT-B/2 \
#         --channel-mult 1,2,3,4 \
#         --attention-resolutions 16,8 \
#         --normalize-matrix statistic/stats_25.npy \
#         --use-diffloss \
#         --c-by-loss-std \
#         --linear-act gate_relu \
#         --norm-type rms \
#         --projector-dim 2048 \
#         --repa-lamb 0.1 \
#         --repa-enc-info 4:dinov2-vit-b \
#         --repa-relu-margin 0.4 \
#         --repa-timesteps full \
#         --denoising-task-rate 0.5 \
#         --repa-mapper repa \
#         --mar-mapper-num-res-blocks 0 \
#         --num-register 0 \
#         --freq-type prev_mlp \
#         --use-rope \
#         --cond-mixing \
#         --vae vae \
#         --fwd ve \
#         --p-mean -1.5 \
#         --p-std 2.0 \
#         --sigma-data 0.5 \
#         --diff-rate 0.7 \
#         --diff-lamb 5 \
#         --c-type edm \
#         --opt radam \
#         --compile \
#         --use-repa \
#         --tau 1.0 \
#         # --resume \
# #         --ckpt-every 1 \
# #         --plot-every 1 \
# #         --model-ckpt /research/cbim/medical/qd66/lct_v2_new/imagenet_256_va/va_DiT_ve_repa_register_0_L_premlp_noot_karras-0.8,1.5_diff_0.70_rho7_ict/checkpoints/0000450.pt \