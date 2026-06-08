#!/usr/bin/env bash
#SBATCH -J gr1_prepare
#SBATCH -p debug
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH -o /data/user/wsong890/shuaizhou/d0/outputs/gr1_lerobot/prepare_%j.out
#SBATCH -e /data/user/wsong890/shuaizhou/d0/outputs/gr1_lerobot/prepare_%j.err

set -euo pipefail

cd /data/user/wsong890/shuaizhou/d0
mkdir -p outputs/gr1_lerobot

ROOT=/data/user/wsong890/shuaizhou/d0/dataset/robot_data/physicalai_groot_x_embodiment_sim
WAN_PATH=/data/user/wsong890/user68/cjy/Motus/pretrained_models
CONDA_ENV=${CONDA_ENV:-/data/user/wsong890/user68/conda_env/groot}
PYTHON=${CONDA_ENV}/bin/python
TASKS=(
  gr1_arms_waist.CanToDrawer
  gr1_arms_waist.CupToDrawer
)

echo "ROOT=${ROOT}"
echo "PYTHON=${PYTHON}"
for task in "${TASKS[@]}"; do
  parquet_count=$(find "${ROOT}/${task}/data" -name '*.parquet' 2>/dev/null | wc -l)
  video_count=$(find "${ROOT}/${task}/videos" -name '*.mp4' 2>/dev/null | wc -l)
  echo "${task}: parquet=${parquet_count} mp4=${video_count}"
  if [[ "${parquet_count}" -eq 0 || "${parquet_count}" -ne "${video_count}" ]]; then
    echo "Task ${task} is incomplete; expected nonzero parquet count equal to mp4 count." >&2
    exit 1
  fi
done

"${PYTHON}" data/lerobot/build_gr1_stats.py \
  --root "${ROOT}" \
  --output "${ROOT}/gr1_action_stats.json" \
  --stats-key gr1_arms_waist \
  --tasks "${TASKS[@]}"

for task in "${TASKS[@]}"; do
  "${PYTHON}" data/lerobot/add_t5_cache_to_lerobot_dataset.py \
    --repo_id "${task}" \
    --root "${ROOT}/${task}" \
    --wan_path "${WAN_PATH}" \
    --device cuda \
    --text_len 512 \
    --t5_folder_name t5_embedding \
    --share_by_instruction
done

echo "GR1 two-task prepare completed."
