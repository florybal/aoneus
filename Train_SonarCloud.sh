#!/usr/bin/env bash
# roda uma reconstrução por cenário
for scene_dir in ./data/SonarCloud/*/*/moved_terrain*; do
    exp_id=$(echo "$scene_dir" | sed 's#\./data/SonarCloud/##; s#/#_#g')
    conf_out="confs/generated/${exp_id}.conf"

    # gera o .conf a partir de um template, substituindo dataset e expID
    sed -e "s|__DATASET__|${scene_dir}|g" \
        -e "s|__EXPID__|${exp_id}|g" \
        confs/template_sonarcloud.conf > "$conf_out"

    python run_sdf.py --conf "$conf_out" --disable_wandb --gpu 0
done