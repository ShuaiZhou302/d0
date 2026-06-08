#!/usr/bin/env python3
"""Inspect one GR1 LeRobot batch without constructing the Motus model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.dataset import collate_fn, create_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gr1_lerobot_finetune_baseline_smoke.yaml")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.common.action_chunk_size = config.common.num_video_frames * config.common.video_action_freq_ratio
    dataset = create_dataset(config, val=args.split == "val")
    loader = DataLoader(
        dataset,
        batch_size=int(config.training.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )
    batch = next(iter(loader))
    if batch is None:
        raise RuntimeError("collate_fn returned None")

    for key in ["first_frame", "video_frames", "initial_state", "action_sequence", "language_embedding"]:
        value = batch.get(key)
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype} finite={torch.isfinite(value.float()).all().item()}")
        else:
            print(f"{key}: type={type(value)}")

    vlm_inputs = batch.get("vlm_inputs")
    if isinstance(vlm_inputs, dict):
        print(f"vlm_inputs: keys={sorted(vlm_inputs.keys())}")


if __name__ == "__main__":
    main()
