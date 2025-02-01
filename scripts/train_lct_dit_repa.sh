export MASTER_PORT=10121
PREFIX=/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/latent_celeb256
NUM_GPUS=$1

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10120 --nproc_per_node=2 train_cm_latent.py \
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 train_cm_latent.py \

CUDA_VISIBLE_DEVICES=1 torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS train_cm_latent.py \
        --exp baseline_repa0.5  \
        --datadir $PREFIX/ \
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
        --global-batch-size $((32*2)) \
        --epochs $((1400*1)) \
        --lr 1e-4 \
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
        --plot-every 5 \
        --use-repa \
        --projector-dim 2048 \
        --enc-type dinov2-vit-b \
        --encoder-depth 4 \
        --repa-lamb 0.5 \
        # --diff-lamb 5 \
        # --use-bf16 \
        
        
        
        # # baseline L/2: 
        #         - FID 6.3 6.4 (ngang diffusion)
        #         -700 epochs (best 675)
        # # baseline B/2: 
        #         - FID 7.6
        #         -700 epochs (best 650)
        # # RELU (thay vi GeLU, ko tot bang) --> dung GeLU

        # # Note: 

        # # epochs: ~700-800
        # # 


        # # Note that:
        # # --normalize-matrix celeb256_stat.npy \ <-- replace this after running extract_stat.py
        # # model cang to hoac cang train lau --> cang de exploding (chi xay ra voi attention)
        
        # # temporal loss 