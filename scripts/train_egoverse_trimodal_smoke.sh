#!/bin/bash
set -euo pipefail

source /share/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/data/user/wsong890/envs/motus}"

TASK="${TASK:-egoverse_trimodal_smoke}"
CONFIG_FILE="${CONFIG_FILE:-configs/egoverse_trimodal_smoke.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-${TASK}}"
MASTER_PORT="${MASTER_PORT:-29521}"
mkdir -p "${OUTPUT_DIR}"

torchrun \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT}" \
  train/train.py \
  --config "${CONFIG_FILE}" \
  --run_name "${TASK}" \
  --report_to tensorboard \
  2>&1 | tee "${OUTPUT_DIR}/train_${TASK}.log"
