# Latent Action VAE

This folder is for reproducing the Motus latent-action generator as closely as the public paper/code allow.

The current main path is paper-aligned:

```text
DPFlow optical flow [T - 1, 2, H, W]
-> flow RGB [T - 1, 3, 256, 256]
-> DC-AE f128c512
-> 4 x 512 tokens
-> lightweight encoder
-> 14D latent action
-> lightweight decoder
-> 4 x 512 reconstructed tokens
-> DC-AE decoder
-> reconstructed flow RGB
```

`baseline_conv_vae.py` contains the earlier simple Conv VAE only as a cheap baseline. It is not the main implementation.

## Files

- `model.py`: DC-AE-token latent action VAE, with `[B, 4, 512] -> z14 -> [B, 4, 512]`.
- `dcae_wrapper.py`: loads public DC-AE checkpoints from EfficientViT/Hugging Face.
- `io.py`: DPFlow XY tensor loading, flow XY to RGB conversion, resizing, and synthetic flow helpers.
- `train.py`: training entrypoint for DC-AE-token latent action VAE.
- `export_latents.py`: exports `[T - 1, 14]` latent action tensors from a trained checkpoint.
- `smoke_test.py`: fake-DC-AE CPU shape test for the paper-aligned architecture.
- `baseline_conv_vae.py`: old simple Conv VAE baseline.
- `third_party/ptlflow`: PTLFlow submodule containing DPFlow.
- `third_party/efficientvit`: EfficientViT submodule containing DC-AE.

## Submodules

On a fresh machine:

```bash
git submodule update --init --recursive latent_action_vae/third_party/ptlflow
git submodule update --init --recursive latent_action_vae/third_party/efficientvit
```

Or clone the main repo with:

```bash
git clone --recurse-submodules <repo-url>
```

## DPFlow

The Motus paper uses DPFlow for optical flow. DPFlow is released inside PTLFlow:

```text
latent_action_vae/third_party/ptlflow/ptlflow/models/dpflow
```

Upstream:

```text
https://github.com/hmorimitsu/ptlflow
```

Pinned commit:

```text
e9cf9bcedd06cba2abe94d6bddfb2ed4f91ac3a2
```

DPFlow dependencies and pretrained weights are not installed/downloaded in this repo yet.

## DC-AE

The Motus paper says flow RGB is compressed by DC-AE into `4 x 512` tokens. The public DC-AE implementation lives in EfficientViT:

```text
latent_action_vae/third_party/efficientvit
```

Upstream:

```text
https://github.com/mit-han-lab/efficientvit
```

Pinned commit:

```text
de7d7733cc0329f391b33f1f459271562ec27bd5
```

The shape-matched checkpoint family is:

```text
dc-ae-f128c512 + 256x256 flow RGB
=> 512 x (256 / 128) x (256 / 128)
=> 512 x 2 x 2
=> 4 x 512 tokens
```

Default model:

```text
mit-han-lab/dc-ae-f128c512-mix-1.0
```

Alternative:

```text
mit-han-lab/dc-ae-f128c512-in-1.0
```

Weights are loaded on the training server through `DCAE_HF.from_pretrained(...)`. They are not committed to this repo.

## Trainability Decision

We use the public DC-AE checkpoint, but **do not freeze it by default**.

Reason: the public checkpoint is trained for natural images, while this task reconstructs flow RGB. The paper describes DC-AE reconstructing flow; since it does not state that DC-AE is frozen, our default is to finetune DC-AE jointly with the lightweight latent encoder/decoder.

There is a `--freeze-dcae` flag for ablations only.

## Flow RGB Convention

The figure shows a colorized flow image, but the paper does not specify the exact color wheel or scale. Current implementation uses a deterministic HSV-style conversion:

```text
angle -> hue
magnitude / max_flow -> value
saturation = 1
RGB [0, 1] -> normalized RGB [-1, 1]
```

This keeps the pipeline paper-aligned while leaving the convention easy to swap if the authors clarify it.

## Smoke Test

This smoke test does not download DC-AE or use real videos. It uses `FakeDCAE` with the same shape contract:

```bash
python -m latent_action_vae.smoke_test
```

Expected checks:

```text
tokens_shape=(16, 4, 512)
latent_shape=(16, 14)
recon_shape=(2, 3, 256, 256)
smoke test passed
```

## Train on Real Flow Files

After extracting DPFlow XY tensors:

```bash
python -m latent_action_vae.train \
  --input-dir /path/to/flow_pt_files \
  --output latent_action_vae/checkpoints/paper_latent_action_vae.pt \
  --dcae-model dc-ae-f128c512-mix-1.0 \
  --input-size 256 \
  --max-flow 20 \
  --steps 1000
```

Expected flow `.pt` formats:

```python
torch.Tensor  # [N, 2, H, W] or [N, H, W, 2]
{"flow": tensor}
{"optical_flow": tensor}
```

## Export Motus Labels

```bash
python -m latent_action_vae.export_latents \
  --checkpoint latent_action_vae/checkpoints/paper_latent_action_vae.pt \
  --input /path/to/optical_flow/example_000.pt \
  --output /path/to/human_ego_pretrain/latent_action_dim14/example_000.pt
```

Output:

```python
torch.Tensor  # [T - 1, 14]
```

## Labeled Robot Alignment

The paper trains with 90% unlabeled data for flow reconstruction and 10% labeled trajectories for weak action supervision. The labeled portion includes task-agnostic data, following AnyPos/Curobo, and standard robot demonstrations.

`model.py` already includes an `action_head` and `latent_action_vae_loss(..., robot_action=...)` hook. The actual mixed dataloader is intentionally not implemented yet because we need to inspect the robot data format on the GPU/data server first.

## Remaining Unknowns

- Exact DPFlow checkpoint used by Motus.
- Exact flow RGB colorization convention.
- Whether the authors froze or finetuned DC-AE.
- Exact lightweight encoder/decoder architecture.
- Exact loss weights for reconstruction, action alignment, token reconstruction, and KL.
- Exact robot labeled datasets and action normalization.

## TODO Before Real Training

These are intentionally not implemented yet because the video/action data and GPU environment are on another server.

- Add `extract_dpflow.py`: video or frame sequence -> DPFlow XY `.pt` with shape `[T - 1, 2, H, W]`.
- Install and validate PTLFlow/DPFlow on the GPU server.
- Download and validate `mit-han-lab/dc-ae-f128c512-mix-1.0`.
- Run a real DC-AE shape test:

```text
flow RGB [B, 3, 256, 256]
-> DC-AE encode
-> [B, 512, 2, 2]
-> tokens [B, 4, 512]
```

- Inspect the real human video data layout and decide whether to cut segments or write an episode+JSON dataset.
- Inspect labeled robot data format before implementing action alignment:

```text
image_t / image_t+1 / action_t
video / action sequence
action dim and normalization
frame-action frequency alignment
```

- Implement mixed training batches:

```text
90% unlabeled flow reconstruction
10% labeled trajectory weak action supervision
```

- Enable `robot_action` in the training loop and tune `action_weight` / `beta`.
- Add export pipeline from trained checkpoint to Motus format:

```text
human_ego_pretrain/latent_action_dim14/*.pt
```

- Add a dataloader smoke test against `data/latent_action/latent_action_dataset.py`.
