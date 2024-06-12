export CUDA_VISIBLE_DEVICES="0" 

torchrun \
    --nnodes=1 \
    --rdzv_endpoint 0.0.0.0:8006 \
    --nproc_per_node=1 \
    vim/train_edm2_encoder.py \
    --exp L_2_linear_block_cpe_dilated_style_EDM2_encoder \
    --model DiM-L/2_Jamba \
    --datadir ./features/celeba256.npy \
    --dataset latent_celeba_256 \
    --global-batch-size 28 \
    --lr 1e-4 \
    --epochs 500 \
    --learn-sigma \
    --pe-type ape \
    --block-type linear \
    --no-lr-decay \
    --ckpt-every 25 \
    # --use_dilated \
    # --resume ./results/L_2_linear_block_cpe_dilated-DiM-L-2/checkpoints/0000250.pt \
    # --using_dct \ # --use_dilated \ --use_wavelet \