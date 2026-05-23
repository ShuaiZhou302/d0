#!/bin/bash
set -euo pipefail

source /share/anaconda3/etc/profile.d/conda.sh
conda activate /data/user/wsong890/envs/motus

TASK="egoverse_vgm"
CONFIG_FILE="configs/egoverse_vgm.yaml"
OUTPUT_DIR="outputs/motus-${TASK}"
mkdir -p "${OUTPUT_DIR}"

torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=29501 \
  train/train.py \
  --config "${CONFIG_FILE}" \
  --run_name "${TASK}_smoke" \
  --report_to tensorboard \
  > "${OUTPUT_DIR}/train_egoverse_vgm_smoke.log" 2>&1

