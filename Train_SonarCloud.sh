#!/usr/bin/env bash

# roda uma reconstrução por cena (_data)

for scene_dir in ./data/SonarCloud/*_data; do

    exp_id=$(basename "$scene_dir")

    conf_out="confs/generated/${exp_id}.conf"

    # gera o .conf a partir de um template,
    # substituindo dataset e expID
    sed \
        -e "s|__DATASET__|${scene_dir}|g" \
        -e "s|__EXPID__|${exp_id}|g" \
        confs/template_sonarcloud.conf > "$conf_out"

    echo "=========================================="
    echo "Treinando: ${exp_id}"
    echo "Dataset:   ${scene_dir}"
    echo "Config:    ${conf_out}"
    echo "=========================================="

    python run_sdf.py \
        --conf "$conf_out" \
        --disable_wandb \
        --gpu 0

done