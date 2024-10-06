export CUDA_VISIBLE_DEVICES="0"

python pytorch_fid/fid_score.py \
    --batch-size 32 \
    --device cuda:0 \
    --dims 2048 \
    ~/repo/LlamaGen/check_celeba256 \
    ~/datasets/celeba_256_jpg

python ~/envs/slack_workflow/running_finished.py
