export CUDA_VISIBLE_DEVICES="0"

torchrun \
    --nnodes=1 \
    --rdzv_endpoint 0.0.0.0:8007 \
    --nproc_per_node=1 vim/sample_ddp.py \
    --ckpt ./results/L_2_linear_block_cpe_dilated_style-DiM-L-2_Jamba/checkpoints/0000500.pt \
    --sample-dir ./sample/ \
    --per-proc-batch-size 50 \
    --num-fid-samples 50000 \
    --num-sampling-steps 250 \
    --global-seed 0 \
    --model DiM-L/2_Jamba \
    --learn-sigma \
    --pe-type ape \
    --block-type linear \
    --eta 0.6

python eval_toolbox/calc_metrics.py \
    --metrics fid50k_full \
    --data ~/repo/edm2/data/real_samples/celeba_256 \
    --mirror 1 \
    --gen_data sample/L_2_linear_block_cpe_dilated_style-DiM-L-2_Jamba/DiM-L-2_Jamba-0000500-size-256-cfg-1-seed-0 \
    --img_resolution 256

python ~/envs/slack_workflow/running_finished.py
