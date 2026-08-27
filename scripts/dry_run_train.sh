#!/usr/bin/env bash
#
# Dry-run do Train_SonarCloud.sh: simula o que seria executado sem rodar treino.
#
set -euo pipefail

TEMPLATE="confs/sonar_cloud/template_sonarcloud.conf"
CONFS_DIR="confs/generated"
FILTER_OBJECT="${1:-}"
mkdir -p "${CONFS_DIR}"

if [[ ! -f "${TEMPLATE}" ]]; then
    echo "[ERRO] Template não encontrado: ${TEMPLATE}"
    exit 1
fi

shopt -s nullglob
SCENES=( data/SonarCloud/*_data/*/moved_terrain* )
shopt -u nullglob

echo "[DRY-RUN] ${#SCENES[@]} cena(s) seriam processadas."
echo

COUNT=0
for scene_dir in "${SCENES[@]}"; do
    obj_name=$(basename "$(dirname "$(dirname "${scene_dir}")")")
    terrain_name=$(basename "$(dirname "${scene_dir}")")
    moved_name=$(basename "${scene_dir}")

    if [[ -n "${FILTER_OBJECT}" && "${obj_name}" != "${FILTER_OBJECT}" ]]; then
        continue
    fi

    exp_id="${obj_name}_${terrain_name}_${moved_name}"
    conf_out="${CONFS_DIR}/${exp_id}.conf"

    # Gera conf e mostra as substituições
    sed \
        -e "s|__DATASET__|${scene_dir}|g" \
        -e "s|__EXPID__|${exp_id}|g" \
        "${TEMPLATE}" > "${conf_out}"

    echo "[${COUNT}] expID=${exp_id}"
    echo "      conf=${conf_out}"
    echo "      dataset=${scene_dir}"
    COUNT=$((COUNT + 1))

    if [[ ${COUNT} -ge 3 ]]; then
        echo "      ... (${#SCENES[@]} total, parando preview)"
        break
    fi
done