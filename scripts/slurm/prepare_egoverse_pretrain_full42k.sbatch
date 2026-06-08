#!/bin/bash
#SBATCH -p acd_u
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH -t 1-00:00:00
#SBATCH -J prep_ego42k
#SBATCH -o /data/user/wsong890/shuaizhou/d0/outputs/motus-egoverse_trimodal_action_frozen_full42k_300k/prepare_%j.out
#SBATCH -e /data/user/wsong890/shuaizhou/d0/outputs/motus-egoverse_trimodal_action_frozen_full42k_300k/prepare_%j.err

set -euo pipefail

cd "${D0_ROOT:-/data/user/wsong890/shuaizhou/d0}"
mkdir -p outputs/motus-egoverse_trimodal_action_frozen_full42k_300k

export CONDA_ENV="${CONDA_ENV:-/data/user/wsong890/envs/motus}"
export WAN_REPO_PATH="${WAN_REPO_PATH:-/data/user/wsong890/user68/cjy/Motus/pretrained_models/Wan2.2-TI2V-5B}"
export CUDA_DEVICES_FOR_T5="${CUDA_DEVICES_FOR_T5:-0,1,2,3,4,5,6,7}"
export MAX_VIDEOS="${MAX_VIDEOS:-0}"

bash scripts/prepare_egoverse_pretrain_current.sh
