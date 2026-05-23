#!/bin/bash
set -euo pipefail

# Full EgoVerse VGM-only pipeline:
# 1. build full train/val manifests
# 2. generate WAN/UMT5 embeddings
# 3. run VGM-only training
#
# Override any variable from the command line, e.g.:
#   MAX_VIDEOS=500 MAX_STEPS=500 bash scripts/run_egoverse_vgm_full.sh

REPO_ROOT="${REPO_ROOT:-/data/user/wsong890/shuaizhou/d0}"
CONDA_ENV="${CONDA_ENV:-/data/user/wsong890/envs/motus}"
CONDA_SETUP="${CONDA_SETUP:-/share/anaconda3/etc/profile.d/conda.sh}"

RAW_ROOT="${RAW_ROOT:-dataset/human_data/egoverse_raw/EgoVerse}"
OUTPUT_ROOT="${OUTPUT_ROOT:-dataset/human_data/egoverse_vgm}"
WAN_REPO_PATH="${WAN_REPO_PATH:-/data/user/wsong890/user68/cjy/Motus/pretrained_models/Wan2.2-TI2V-5B}"
CONFIG_FILE="${CONFIG_FILE:-configs/egoverse_vgm.yaml}"

MAX_VIDEOS="${MAX_VIDEOS:-0}"
MAX_STEPS="${MAX_STEPS:-500}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29511}"
RUN_NAME="${RUN_NAME:-egoverse_vgm_full_${MAX_STEPS}steps}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/motus-egoverse-vgm-full}"

SKIP_MANIFEST="${SKIP_MANIFEST:-0}"
SKIP_UMT5="${SKIP_UMT5:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

source "${CONDA_SETUP}"
conda activate "${CONDA_ENV}"
cd "${REPO_ROOT}"

mkdir -p "${OUTPUT_DIR}" configs/generated

echo "[EgoVerse VGM] repo: ${REPO_ROOT}"
echo "[EgoVerse VGM] raw root: ${RAW_ROOT}"
echo "[EgoVerse VGM] output root: ${OUTPUT_ROOT}"
echo "[EgoVerse VGM] run name: ${RUN_NAME}"

if [[ "${SKIP_MANIFEST}" != "1" ]]; then
  MANIFEST_ARGS=(
    --raw-root "${RAW_ROOT}"
    --output-root "${OUTPUT_ROOT}"
  )
  if [[ "${MAX_VIDEOS}" != "0" ]]; then
    MANIFEST_ARGS+=(--max-videos "${MAX_VIDEOS}")
  fi

  python -m data.egoverse.build_egoverse_vgm_manifest "${MANIFEST_ARGS[@]}" \
    2>&1 | tee "${OUTPUT_DIR}/build_manifest.log"

  wc -l \
    "${OUTPUT_ROOT}/manifests/train.jsonl" \
    "${OUTPUT_ROOT}/manifests/val.jsonl" \
    | tee "${OUTPUT_DIR}/manifest_counts.txt"
fi

if [[ "${SKIP_UMT5}" != "1" ]]; then
  python -m data.egoverse.generate_egoverse_umt5 \
    --manifest "${OUTPUT_ROOT}/manifests/train.jsonl" \
    --manifest "${OUTPUT_ROOT}/manifests/val.jsonl" \
    --wan-repo-path "${WAN_REPO_PATH}" \
    --cuda-devices "${CUDA_DEVICES}" \
    2>&1 | tee "${OUTPUT_DIR}/generate_umt5.log"
fi

GENERATED_CONFIG="configs/generated/egoverse_vgm_${RUN_NAME}.yaml"
python - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("${CONFIG_FILE}")
cfg.dataset.train_manifest = "${OUTPUT_ROOT}/manifests/train.jsonl"
cfg.dataset.val_manifest = "${OUTPUT_ROOT}/manifests/val.jsonl"
cfg.dataset.max_samples = None if int("${MAX_SAMPLES}") <= 0 else int("${MAX_SAMPLES}")
cfg.training.max_steps = int("${MAX_STEPS}")
cfg.training.batch_size = int("${BATCH_SIZE}")
cfg.system.num_workers = int("${NUM_WORKERS}")
cfg.logging.report_to = "tensorboard"
OmegaConf.save(cfg, "${GENERATED_CONFIG}")
print("wrote ${GENERATED_CONFIG}")
PY

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    train/train.py \
    --config "${GENERATED_CONFIG}" \
    --run_name "${RUN_NAME}" \
    --report_to tensorboard \
    2>&1 | tee "${OUTPUT_DIR}/train_${RUN_NAME}.log"
fi

echo "[EgoVerse VGM] done"
