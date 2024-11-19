export MASTER_PORT=10121

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=4 train_cm_latent_img512_edm2.py \
        --exp edm2_xs_cauchy_no_grad_norm_bs128 \
        --datadir ./dataset/ \
        --dataset latent_imagenet512 \
        --results-dir ./results/ \
        --image-size 64 \
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
        --global-batch-size $((64*4)) \
        --epochs $((140*1)) \
        --lr 1e-4 \
        --num-sampling 8 \
        --ict \
        --max-grad-norm 100.0 \
        --model-type EDM2-XS \
        --normalize-matrix imagenet512_stat.npy \
        --use-diffloss \
        --ot-hard \
        --c-by-loss-std \
        --optim Adam \
        --optim-eps 1e-4 \
        # --resume \

python ~/envs/slack_workflow/running_finished.py        
