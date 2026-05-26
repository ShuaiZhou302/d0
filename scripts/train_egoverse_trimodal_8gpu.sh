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

mkdir -p "${OUTPUT_DIR}"

echo "Task: ${TASK}"
echo "Config: ${CONFIG_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NPROC_PER_NODE}"
echo "DeepSpeed: ${DEEPSPEED_CONFIG}"
echo "Run name: ${RUN_NAME}"

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
  --report_to tensorboard \
  > "${OUTPUT_DIR}/train_${TASK}.log" 2>&1
