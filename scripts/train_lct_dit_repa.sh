export OMP_NUM_THREADS=8
# Generate a random port number between 10000 and 20000
MASTER_PORT=$((10000 + RANDOM % 10000))
DATASET=/lustre/scratch/client/movian/research/users/anhnd72/datasets/LCT/latent_celeb256
NUM_GPUS=$1

BATCH_SIZE=64
LR=1e-4
DEPTH=4
REPALAMB=0.5
DIFFLAMB=1.0
ENCTYPE=dinov2-vit-b
EPOCHS=1400
GRAD_NORM=100.0
MODEL_TYPE=DiT-B/2
# MODEL_TYPE=LightningDiT-B/2
# START_SCALES=10 # 10 20 40 80 160 320 640
START_SCALES=10

# CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:10120 --nproc_per_node=2 train_cm_latent.py \
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 train_cm_latent.py \

# CUDA_VISIBLE_DEVICES=4,5,6,7 
        # --exp REPA${REPALAMB}_DIFF${DIFFLAMB}_DEPTH${DEPTH}_LR${LR}_BS${BATCH_SIZE}_ENCTYPE${ENCTYPE}_EPOCHS${EPOCHS}_GRADNORM${GRAD_NORM}_${MODEL_TYPE}_START_SCALES${START_SCALES}_scale4insteadof2  \
CUDA_VISIBLE_DEVICES=2 torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT train_cm_latent.py \
        --exp NEWBASELINE_REPA${REPALAMB}_DIFF${DIFFLAMB}_DEPTH${DEPTH}_LR${LR}_BS${BATCH_SIZE}_ENCTYPE${ENCTYPE}_EPOCHS${EPOCHS}_GRADNORM${GRAD_NORM}_${MODEL_TYPE}_START_SCALES${START_SCALES}  \
        --datadir $DATASET/ \
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
        --start-scales $START_SCALES \
        --end-scales 640 \
        --noise-sampler ict \
        --global-batch-size $((BATCH_SIZE)) \
        --epochs $EPOCHS \
        --lr $LR \
        --num-sampling 8 \
        --ict \
        --max-grad-norm $GRAD_NORM \
        --model-type $MODEL_TYPE \
        --normalize-matrix $DATASET/stats.npy \
        --use-diffloss \
        --ot-hard \
        --c-by-loss-std \
        --plot-every 1 \
        --num-workers 16 \
        --projector-dim 2048 \
        --enc-type $ENCTYPE \
        --encoder-depth $DEPTH \
        --repa-lamb $REPALAMB \
        --diff-lamb $DIFFLAMB \
        --z_dims 768 \
        --ckpt-every 100 \
        --uncond-network \
        --use-repa \
        # --use-sigmoid-attention \
        # --normalize-matrix celeb256_stat.npy \
        # 
        # --diff-lamb 5 \
        # --use-bf16 \
        # --projector-dim 2048 \

        
        
        
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