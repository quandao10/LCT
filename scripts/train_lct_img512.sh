export MASTER_PORT=10121

CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --rdzv_endpoint 0.0.0.0:$MASTER_PORT --nproc_per_node=1 train_cm_latent_img512_edm2.py \
        --exp edm2_xs_cauchy_no_grad_norm_bs128_lr1e-3_test2 \
        --datadir ./dataset/ \
        --dataset latent_imagenet512 \
        --results-dir ./results/ \
        --image-size 16 \
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
        --global-batch-size $((640*1)) \
        --epochs $((140*1)) \
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
        --normalize-matrix imagenet512_stat.npy \
        --use-diffloss \
        --c-by-loss-std \
        --optim RAdam \
        --optim-eps 1e-4 \
        --last-norm-type non-scaling-layer-norm \
        --block-norm-type non-scaling-layer-norm \
        # --ot-hard \
        # --resume \
        # --plot-every 10 \
        # --edm2-pretrained \

python ~/envs/slack_workflow/running_finished.py        
