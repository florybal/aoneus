#!/usr/bin/env bash
#
# Treina uma reconstrução NeuS para cada cena do SonarCloud.
#
# Para cada combinação <objeto>/<terreno>/<posição> em
#   data/SonarCloud/*_data/*/moved_terrain*
# gera um .conf a partir de confs/sonar_cloud/template_sonarcloud.conf
# e roda run_sdf.py no ambiente conda `aoneus_5090`.
#
# Uso:
#   ./Train_SonarCloud.sh                  # treina todas as cenas
#   ./Train_SonarCloud.sh sphere_data      # treina apenas sphere_data
#   GPU=1 ./Train_SonarCloud.sh            # usa GPU 1
#
set -euo pipefail

# -------------------------------------------------
# Configurações
# -------------------------------------------------
TEMPLATE="confs/sonar_cloud/template_sonarcloud.conf"
CONFS_DIR="confs/generated"
GPU="${GPU:-0}"
CONDA_ENV="${CONDA_ENV:-aoneus_5090}"
FILTER_OBJECT="${1:-}"        # se informado, filtra por nome do objeto (ex: sphere_data)

mkdir -p "${CONFS_DIR}"

# -------------------------------------------------
# Verificações iniciais
# -------------------------------------------------
if [[ ! -f "${TEMPLATE}" ]]; then
    echo "[ERRO] Template não encontrado: ${TEMPLATE}" >&2
    echo "       Esperado em confs/sonar_cloud/template_sonarcloud.conf" >&2
    exit 1
fi

# -------------------------------------------------
# Loop principal
# -------------------------------------------------
# Cada cena válida é: <objeto>/<terreno>/<posição>
shopt -s nullglob
SCENES=( data/SonarCloud/*_data/*/moved_terrain* )
shopt -u nullglob

if [[ ${#SCENES[@]} -eq 0 ]]; then
    echo "[ERRO] Nenhuma cena encontrada em data/SonarCloud/*_data/*/moved_terrain*" >&2
    exit 1
fi

echo "[INFO] ${#SCENES[@]} cena(s) encontrada(s)."
echo

RUN_COUNT=0
for scene_dir in "${SCENES[@]}"; do
    # Extrai nome do objeto (ex: sphere_data)
    obj_name=$(basename "$(dirname "$(dirname "${scene_dir}")")")
    terrain_name=$(basename "$(dirname "${scene_dir}")")
    moved_name=$(basename "${scene_dir}")

    # Filtro opcional por objeto
    if [[ -n "${FILTER_OBJECT}" && "${obj_name}" != "${FILTER_OBJECT}" ]]; then
        continue
    fi

    # expID único, ex: sphere_data_no_terrain_moved_terrain0
    exp_id="${obj_name}_${terrain_name}_${moved_name}"
    conf_out="${CONFS_DIR}/${exp_id}.conf"

    # Gera o .conf substituindo placeholders
    sed \
        -e "s|__DATASET__|${scene_dir}|g" \
        -e "s|__EXPID__|${exp_id}|g" \
        "${TEMPLATE}" > "${conf_out}"

    echo "=========================================="
    echo "[${RUN_COUNT}] Objeto: ${obj_name}"
    echo "     Terreno: ${terrain_name}"
    echo "     Posição: ${moved_name}"
    echo "     Dataset: ${scene_dir}"
    echo "     Config : ${conf_out}"
    echo "     ExpID  : ${exp_id}"
    echo "=========================================="

    # Roda o treino. Usa conda run para garantir o env correto.
    conda run -n "${CONDA_ENV}" python run_sdf.py \
        --conf "${conf_out}" \
        --disable_wandb \
        --gpu "${GPU}"

    RUN_COUNT=$((RUN_COUNT + 1))
    echo
done

echo "=========================================="
echo "[OK] ${RUN_COUNT} treinamento(s) concluído(s)."
echo "=========================================="