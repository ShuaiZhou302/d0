#!/bin/bash
set -euo pipefail

source /share/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/data/user/wsong890/envs/motus}"

TASK="${TASK:-egoverse_trimodal_action_frozen}"
CONFIG_FILE="${CONFIG_FILE:-configs/egoverse_trimodal_pretrain_8gpu.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
MASTER_PORT="${MASTER_PORT:-29531}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/zero2.json}"
RUN_NAME="${RUN_NAME:-${TASK}_partial23k}"
REPORT_TO="${REPORT_TO:-all}"
WANDB_PROJECT="${WANDB_PROJECT:-d0_egoverse_pretrain}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/wandb"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"

echo "Task: ${TASK}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NPROC_PER_NODE}"
echo "DeepSpeed: ${DEEPSPEED_CONFIG}"
echo "Run name: ${RUN_NAME}"
echo "Logging: ${REPORT_TO} (WANDB_MODE=${WANDB_MODE}, WANDB_DIR=${WANDB_DIR})"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  train/train.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --config "${CONFIG_FILE}" \
  --run_name "${RUN_NAME}" \
  --report_to "${REPORT_TO}" \
  --wandb_project "${WANDB_PROJECT}" \
  > "${OUTPUT_DIR}/train_${TASK}.log" 2>&1
