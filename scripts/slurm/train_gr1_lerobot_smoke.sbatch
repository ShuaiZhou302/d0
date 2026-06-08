#!/usr/bin/env bash
#SBATCH -J gr1_smoke
#SBATCH -p debug
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /data/user/wsong890/shuaizhou/d0/outputs/gr1_lerobot/smoke_%j.out
#SBATCH -e /data/user/wsong890/shuaizhou/d0/outputs/gr1_lerobot/smoke_%j.err

set -euo pipefail

cd /data/user/wsong890/shuaizhou/d0
mkdir -p outputs/gr1_lerobot

export CONFIG_FILE="${CONFIG_FILE:-configs/gr1_lerobot_finetune_baseline_smoke.yaml}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-/data/user/wsong890/shuaizhou/d0/outputs/gr1_lerobot/wandb}"
export TOKENIZERS_PARALLELISM=false
export CONDA_ENV="${CONDA_ENV:-/data/user/wsong890/user68/conda_env/groot}"
export PYTHON="${CONDA_ENV}/bin/python"
export PYTHONPATH="/data/user/wsong890/shuaizhou/d0:${PYTHONPATH:-}:/data/user/wsong890/envs/motus/lib/python3.10/site-packages"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "PYTHON=${PYTHON}"

"${PYTHON}" data/lerobot/build_gr1_stats.py \
  --root /data/user/wsong890/shuaizhou/d0/dataset/robot_data/physicalai_groot_x_embodiment_sim \
  --output /data/user/wsong890/shuaizhou/d0/dataset/robot_data/physicalai_groot_x_embodiment_sim/gr1_action_stats.json \
  --stats-key gr1_arms_waist \
  --tasks gr1_arms_waist.CanToDrawer gr1_arms_waist.CupToDrawer

"${PYTHON}" train/train.py --config "${CONFIG_FILE}"
