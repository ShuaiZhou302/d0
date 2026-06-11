#!/bin/bash
#SBATCH -p acd_u
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH -t 4-00:00:00
#SBATCH -J egodex_300k
#SBATCH -o /data/user/wsong890/shuaizhou/d0/outputs/motus-egodex_trimodal_action_frozen_full_300k/slurm_%j.out
#SBATCH -e /data/user/wsong890/shuaizhou/d0/outputs/motus-egodex_trimodal_action_frozen_full_300k/slurm_%j.err

set -euo pipefail

cd "${D0_ROOT:-/data/user/wsong890/shuaizhou/d0}"
mkdir -p outputs/motus-egodex_trimodal_action_frozen_full_300k

export CONDA_ENV="${CONDA_ENV:-/data/user/wsong890/envs/motus}"
export TASK="${TASK:-egodex_from_motus_action_frozen_full_300k}"
export CONFIG_FILE="${CONFIG_FILE:-configs/egodex_trimodal_pretrain_full_300k.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-egodex_trimodal_action_frozen_full_300k}"
export MASTER_PORT="${MASTER_PORT:-29547}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
export RUN_NAME="${RUN_NAME:-egodex_from_motus_action_frozen_full_300k}"
export REPORT_TO="${REPORT_TO:-all}"
export WANDB_PROJECT="${WANDB_PROJECT:-d0_egodex_pretrain}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"

bash scripts/train_egoverse_trimodal_8gpu.sh
