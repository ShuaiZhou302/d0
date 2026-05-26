#!/bin/bash
set -euo pipefail

source /share/anaconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/data/user/wsong890/envs/motus}"

D0_ROOT="${D0_ROOT:-/data/user/wsong890/shuaizhou/d0}"
WAN_REPO_PATH="${WAN_REPO_PATH:-/data/user/wsong890/user68/cjy/Motus/pretrained_models/Wan2.2-TI2V-5B}"
CUDA_DEVICES_FOR_T5="${CUDA_DEVICES_FOR_T5:-0,1,2,3,4,5,6,7}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"

cd "${D0_ROOT}"
export PYTHONPATH="${D0_ROOT}:${PYTHONPATH:-}"

python data/egoverse/build_egoverse_vgm_manifest.py \
  --raw-root dataset/human_data/egoverse_raw/EgoVerse \
  --output-root dataset/human_data/egoverse_vgm \
  --train-ratio 0.95 \
  --seed 0 \
  --num-video-frames 8 \
  --global-downsample-rate 2 \
  --min-confidence 0.5 \
  --max-videos "${MAX_VIDEOS}"

python data/egoverse/generate_egoverse_umt5.py \
  --manifest dataset/human_data/egoverse_vgm/manifests/train.jsonl \
  --manifest dataset/human_data/egoverse_vgm/manifests/val.jsonl \
  --wan-repo-path "${WAN_REPO_PATH}" \
  --cuda-devices "${CUDA_DEVICES_FOR_T5}"
