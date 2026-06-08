#!/usr/bin/env bash
#SBATCH -J gr1_full24
#SBATCH -p acd_u
#SBATCH -N 1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=4-00:00:00
#SBATCH -o /data/user/wsong890/shuaizhou/d0/outputs/gr1_lerobot_full24/slurm_%j.out
#SBATCH -e /data/user/wsong890/shuaizhou/d0/outputs/gr1_lerobot_full24/slurm_%j.err

set -euo pipefail

cd /data/user/wsong890/shuaizhou/d0
mkdir -p outputs/gr1_lerobot_full24

export CONDA_ENV="${CONDA_ENV:-/data/user/wsong890/envs/motus}"
export TASK="${TASK:-gr1_full24}"
export CONFIG_FILE="${CONFIG_FILE:-configs/gr1_lerobot_finetune_ours_full24_100k.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/gr1_lerobot_full24}"
export MASTER_PORT="${MASTER_PORT:-29555}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
export RUN_NAME="${RUN_NAME:-${TASK}_${SLURM_JOB_ID}}"
export REPORT_TO="${REPORT_TO:-all}"
export WANDB_PROJECT="${WANDB_PROJECT:-d0_gr1_downstream}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/data/user/wsong890/shuaizhou/d0:${PYTHONPATH:-}"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "HOST=$(hostname)"
echo "RUN_NAME=${RUN_NAME}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

bash scripts/train_egoverse_trimodal_8gpu.sh
