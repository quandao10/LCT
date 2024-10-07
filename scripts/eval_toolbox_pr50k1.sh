export CUDA_VISIBLE_DEVICES="3"

for epoch in 1675
do
    python eval_toolbox/calc_metrics.py \
        --metrics pr50k3_full \
        --data ../real_samples/lsun/ \
        --mirror 1 \
        --gen_data ./generated_samples/latent_church256_flip/explsun_best_setting_2_resume_more_ckpt_ep000${epoch} \
        --img_resolution 256
done

python ~/envs/slack_workflow/running_finished.py
