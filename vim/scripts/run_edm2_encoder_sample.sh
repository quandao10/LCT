export CUDA_VISIBLE_DEVICES="0"

# for EPOCH in 25 50 75
# do
#     torchrun \
#         --nnodes=1 \
#         --rdzv_endpoint 0.0.0.0:8007 \
#         --nproc_per_node=1 vim/sample_ddp_edm2_encoder.py \
#         --ckpt ./results/L_2_linear_block_cpe_dilated_style-DiM-L-2_Jamba_EDM2-norm-celeba256-pixelwise-DiM-L-2_Jamba/checkpoints/00000$EPOCH.pt \
#         --sample-dir ./sample/ \
#         --per-proc-batch-size 50 \
#         --num-fid-samples 2000 \
#         --num-sampling-steps 100 \
#         --global-seed 0 \
#         --model DiM-L/2_Jamba \
#         --learn-sigma \
#         --pe-type ape \
#         --block-type linear \
#         --eta 0.6

#     python eval_toolbox/calc_metrics.py \
#         --metrics fid2k_full \
#         --data ~/repo/edm2/data/real_samples/celeba_256 \
#         --mirror 1 \
#         --gen_data sample/L_2_linear_block_cpe_dilated_style-DiM-L-2_Jamba_EDM2-norm-celeba256-pixelwise-DiM-L-2_Jamba/DiM-L-2_Jamba-00000$EPOCH-size-256-cfg-1-seed-0 \
#         --img_resolution 256
# done

for EPOCH in 200 225 250
do
    torchrun \
        --nnodes=1 \
        --rdzv_endpoint 0.0.0.0:8007 \
        --nproc_per_node=1 vim/sample_ddp_edm2_encoder.py \
        --ckpt ./results/L_2_linear_block_cpe_dilated_style-DiM-L-2_Jamba_EDM2-norm-celeba256-pixelwise-DiM-L-2_Jamba/checkpoints/0000$EPOCH.pt \
        --sample-dir ./sample/ \
        --per-proc-batch-size 50 \
        --num-fid-samples 10000 \
        --num-sampling-steps 100 \
        --global-seed 0 \
        --model DiM-L/2_Jamba \
        --learn-sigma \
        --pe-type ape \
        --block-type linear \
        --eta 0.6

    python eval_toolbox/calc_metrics.py \
        --metrics fid10k_full \
        --data ~/repo/edm2/data/real_samples/celeba_256 \
        --mirror 1 \
        --gen_data sample/L_2_linear_block_cpe_dilated_style-DiM-L-2_Jamba_EDM2-norm-celeba256-pixelwise-DiM-L-2_Jamba/DiM-L-2_Jamba-0000$EPOCH-size-256-cfg-1-seed-0 \
        --img_resolution 256
done

python ~/envs/slack_workflow/running_finished.py
