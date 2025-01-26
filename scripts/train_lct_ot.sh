# # # export MASTER_PORT=10120

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10112 --nproc_per_node=1 train_cm_latent.py \
        --exp 700ep_L_relu_eps1e-4_unnormalized_linear_norm_pixel_norm_nogradnorm  \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
        --dataset latent_celeb256 \
        --results-dir /research/cbim/medical/qd66/lct_v2/ \
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
        --global-batch-size $((48*1)) \
        --epochs $((700*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100.0 \
        --model-type DiT-L/2 \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix celeb256_stat.npy \
        --use-diffloss \
        --ot-hard \
        --diff-lamb 5 \
        --c-by-loss-std \
        --ckpt-every 5 \
        --save-content-every 5 \
        --plot-every 5 \
        --eps 1e-4 \
        --linear-act relu \
        --wo-norm \
        --attn-type flash \
        --model-ckpt /research/cbim/medical/qd66/lct_v2/latent_celeb256/celeb_dit_best_setting_700ep_L_relu_eps1e-4_unnormalized_linear_norm/checkpoints/0000595.pt
        --resume


        
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10130 --nproc_per_node=4 train_cm_latent.py \
        --exp imagenet_dit_best_setting_700ep_L_gelu_soap  \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
        --dataset latent_imagenet256 \
        --results-dir /research/cbim/medical/qd66/lct_v2/ \
        --image-size 32 \
        --num-in-channels 4 \
        --num-classes 1000 \
        --weight-schedule ict \
        --loss-norm cauchy \
        --target-ema-mode adaptive \
        --start-ema 0.95 \
        --scale-mode progressive \
        --start-scales 10 \
        --end-scales 640 \
        --noise-sampler ict \
        --global-batch-size $((75*4)) \
        --epochs $((700*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --num-channels 128 \
        --num-head-channels 64 \
        --num-res-blocks 4 \
        --resblock-updown \
        --ict \
        --max-grad-norm 100.0 \
        --model-type DiT-L/2 \
        --channel-mult 1,2,3,4 \
        --attention-resolutions 16,8 \
        --normalize-matrix latent_imagenet256_stat.npy \
        --use-diffloss \
        --diff-lamb 10 \
        --c-by-loss-std \
        --ckpt-every 5 \
        --save-content-every 5 \
        --plot-every 1 \
        --eps 1e-4 \
        --linear-act gelu \
        --attn-type flash \
        --cfg-scale 1.0 \
        --wo-norm \
#         --resume \



CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10123 --nproc_per_node=1 train_cm_latent.py \
        --exp celeb_dit_best_setting_700ep_B_relu_small_lr  \
        --datadir /research/cbim/vast/qd66/workspace/dataset/ \
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
        --no-scale
        # --resume