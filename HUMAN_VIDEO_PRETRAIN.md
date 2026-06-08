# Human Video Continued Pretraining

This document records the current EgoVerse human-video pretraining setup and the downstream GR1 posttraining plan.

## Purpose

We use annotated egocentric human videos to continue pretraining Motus, then evaluate whether this continued pretraining improves robot downstream posttraining.

Important bug fix: EgoVerse pretraining must start from the original Motus checkpoint. Earlier runs only initialized WAN/VLM backbones and did not load `/data/user/wsong890/user68/cjy/Motus/pretrained_models/Motus`. The current code fixes this by adding a `pretrain.checkpoint_path` path for `training_mode: pretrain`.

## EgoVerse Pretraining

Config:

```text
configs/egoverse_trimodal_pretrain_full42k_300k.yaml
```

Launch script:

```text
scripts/slurm/train_egoverse_trimodal_full42k_300k.sbatch
```

Initialization:

```text
pretrain.checkpoint_path: /data/user/wsong890/user68/cjy/Motus/pretrained_models/Motus
training_mode: pretrain
```

Frozen modules:

```text
VLM: frozen
Action expert: frozen
WAN VAE encoder: no_grad in the training step
```

Trainable modules:

```text
WAN/video diffusion path
Understanding expert
Shared / tri-modal attention parameters that remain trainable
```

Data:

```text
dataset/human_data/egoverse_vgm/manifests/train.jsonl
dataset/human_data/egoverse_vgm/manifests/val.jsonl
```

The split is video-level, not segment-level. On hpc3 at the time of this note:

```text
train: 385600 segments, 40045 videos
val:    19766 segments,  2108 videos
train/val video overlap: 0
```

Debug EgoVerse videos are currently kept in the manifests and are allowed for this run.

Batch and clip format:

```text
batch_size: 6 per GPU
8 GPUs -> global batch 48 clips/step
num_video_frames: 8
global_downsample_rate: 2
```

Each dataset sample is one segment-level video clip:

```text
first_frame:       [3, 384, 320]
video_frames:      [8, 3, 384, 320]
language_embedding: precomputed UMT5/WAN embedding for the segment instruction
vlm_inputs:        first frame + instruction tokens for the VLM/understanding path
action_sequence:   zeros, [8, 14]
initial_state:     not used
```

The model predicts the future frames at:

```text
t + 2, t + 4, ..., t + 16
```

Loss:

```text
total_loss = video_loss_weight * video_loss
video_loss_weight = 1.0
action_loss_weight = 0.0
```

Validation:

```text
val_interval: 50000
```

Validation uses the EgoVerse val manifest to monitor human-video prediction quality. Downstream claims should still be based on robot success-rate evaluation, not pretraining loss alone.

## Checkpoint Loading Fix

The code now supports two partial-load paths:

```text
pretrain.checkpoint_path
finetune.checkpoint_path
```

For continued human-video pretraining:

```text
training_mode: pretrain
pretrain.checkpoint_path: original Motus checkpoint
```

The loader imports shape-compatible Motus weights and refuses to continue if no `vlm_model.*` tensors are loaded. This prevents accidentally training with a randomly initialized VLM.

For GR1 posttraining:

```text
training_mode: finetune
finetune.checkpoint_path: baseline or EgoVerse-pretrained checkpoint
```

GR1 uses a 44D action/state interface. Therefore the 14D action I/O layers from Motus are skipped or shape-mismatch skipped, and the GR1-compatible 44D action/state layers are newly initialized and trained.

## GR1 Downstream A/B

Configs:

```text
configs/gr1_lerobot_finetune_baseline_full24_150k.yaml
configs/gr1_lerobot_finetune_ours_full24_150k.yaml
```

Launch script:

```text
scripts/slurm/train_gr1_lerobot_full24_8gpu.sbatch
```

Baseline:

```text
finetune.checkpoint_path: /data/user/wsong890/user68/cjy/Motus/pretrained_models/Motus
```

Ours:

```text
finetune.checkpoint_path: /data/user/wsong890/shuaizhou/d0/checkpoints/egoverse_trimodal_pretrain_full42k_300k/egoverse_from_motus_action_frozen_full42k_300k/checkpoint_step_300000
```

GR1 posttraining setup:

```text
action_dim: 44
state_dim: 44
num_video_frames: 8
video_action_freq_ratio: 6
batch_size: 4 per GPU
8 GPUs -> global batch 32 clips/step
max_steps: 150000
VLM: frozen
Action expert: trainable
action_loss_weight: 1.0
video_loss_weight: 1.0
```

The intended comparison is:

```text
Baseline: original Motus -> GR1 posttraining
Ours:     original Motus -> EgoVerse continued pretraining -> GR1 posttraining
```

Both runs must use the same GR1 tasks, split, seed, batch size, learning rate, finetuning steps, and evaluation protocol.

## Current Downstream Evaluation Target

The GR1 full24 configs use the 24 `gr1_arms_waist.*` tasks under:

```text
/data/user/wsong890/shuaizhou/d0/dataset/robot_data/physicalai_groot_x_embodiment_sim
```

The first meaningful downstream metric should be simulator success rate:

```text
per-task success rate
average success rate
Ours - Baseline at the same finetuning step
```

Offline video/action losses can be used as smoke diagnostics, but they should not replace simulator success-rate evaluation.
