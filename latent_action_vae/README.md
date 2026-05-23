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

## Paper-Confirmed Structure

The Motus paper specifies this latent-action path:

```text
DPFlow optical flow
-> flow RGB
-> DC-AE reconstruction module
-> four 512-dimensional tokens
-> lightweight encoder
-> 14D latent action
```

The paper also writes the alignment term as matching real actions to the
predicted action/latent action. It does **not** describe a separate `14D -> 14D`
action head after the latent. Therefore `model.py` now uses the encoder mean
`mu` directly as `pred_action` for action supervision and export. The sampled
`z` is used for the VAE reconstruction path during training.

One detail is still not public: the paper does not say whether `a_pred` in the
alignment loss is the sampled latent `z` or a deterministic VAE mean. We use
`mu` because it gives a stable latent-action label at export time and avoids
training the action alignment target against sampling noise.

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

This is an implementation choice, not a paper-confirmed detail.

## Flow RGB Convention

The figure shows a colorized flow image, but the paper does not specify the exact color wheel or scale. Current implementation uses a deterministic HSV-style conversion:

```text
angle -> hue
magnitude / max_flow -> value
saturation = 1
RGB [0, 1] -> normalized RGB [-1, 1]
```

This keeps the pipeline paper-aligned while leaving the convention easy to swap if the authors clarify it.

This is an implementation choice, not a paper-confirmed detail.

## Lightweight Encoder/Decoder

The paper only says "lightweight encoder" and does not publish the exact module.
Current v0 uses a small MLP with `Linear -> LayerNorm -> SiLU` blocks to map:

```text
[B, 4, 512] -> flatten [B, 2048] -> mu/logvar [B, 14]
[B, 14] -> [B, 2048] -> [B, 4, 512]
```

This is an implementation choice, not a paper-confirmed detail.

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

`latent_action_vae_loss(..., robot_action=...)` applies the alignment term
directly between the real 14D robot action and `outputs["pred_action"]`, where
`pred_action == mu`.

## Task-Agnostic Robot Data

The Motus paper says task-agnostic data follows AnyPos: use cuRobo to randomly
sample the target robot action space and collect image-action pairs. AnyPos
reports Mobile ALOHA with 14D joint-position control, three RGB cameras, and
610k task-agnostic image-action pairs collected automatically.

As of this repo version, we do not have a public downloaded AnyPos task-agnostic
dataset in the project. cuRobo is a motion-generation/IK/collision-checking
library, not a dataset by itself. To reproduce this part exactly, we need either:

- an official AnyPos/Motus task-agnostic image-action release, if the authors
  share it;
- or a local generation pipeline: target robot model + camera setup + cuRobo
  sampling/planning + simulator or real robot renderer/logger to save
  `image_t, image_t+1, action_t`.

For v0, `aloha_preprocessed` standard demonstrations provide the labeled
robot-action supervision. That covers the "standard robot demonstrations" part
of Motus, but not the AnyPos-style task-agnostic portion yet.

## Remaining Unknowns

- Exact DPFlow checkpoint used by Motus.
- Exact flow RGB colorization convention.
- Whether the authors froze or finetuned DC-AE.
- Exact lightweight encoder/decoder architecture.
- Exact loss weights for reconstruction, action alignment, token reconstruction, and KL.
- Exact robot labeled datasets and action normalization.
- Whether the action alignment term uses VAE `mu` or sampled `z`; this repo uses `mu`.

## TODO Before Real Training

These are intentionally not implemented yet because the video/action data and GPU environment are on another server.

- Continue HPC setup:

```text
repo: /data/user/wsong890/shuaizhou/d0
env:  /data/user/wsong890/envs/shuai_d0
tmux: motus_latent_shuai
```

- Current HPC status:

```text
shuai_d0 cloned from motus env
torch: 2.7.1+cu126
CPU fake-DC-AE smoke test passes
PTLFlow and EfficientViT submodules copied from local and pinned correctly
timm installed from local wheel
onnx installed from local wheel
PySocks and socksio installed from local wheels for small HF/httpx checks
DC-AE checkpoint stored under this repo:
  latent_action_vae/checkpoints/dc-ae-f128c512-mix-1.0/
GPU DC-AE shape smoke passed on debug node:
  dcae_latent (1, 512, 2, 2)
  tokens      (1, 4, 512)
  mu/logvar   (1, 14)
```

- DC-AE loading notes:

```text
EfficientViT package imports unrelated optional modules while loading DC-AE.
The wrapper now direct-loads EfficientViT's dc_ae.py to avoid SAM/seg/export extras.
onnxsim is shimmed in dcae_wrapper.py because it is only needed for ONNX export.
```

  This does not change the latent-action VAE architecture; it only prevents
  unused EfficientViT extras from blocking DC-AE.

- Load the local DC-AE checkpoint:

```bash
conda run -p /data/user/wsong890/envs/shuai_d0 \
  python -c "from latent_action_vae.dcae_wrapper import load_dcae; p='latent_action_vae/checkpoints/dc-ae-f128c512-mix-1.0'; m=load_dcae(p, device='cpu'); print(type(m).__name__); print(m.spatial_compression_ratio)"
```

- Large files should be downloaded locally and transferred to the cluster:

```text
download locally -> scp to HPC /tmp or repo third_party -> install/extract there
```

- DPFlow extraction is implemented through PTLFlow only. OpenCV flow is not a
  supported backend in this pipeline.
- PTLFlow/DPFlow is installed and validated on the GPU server.
- `mit-han-lab/dc-ae-f128c512-mix-1.0` is stored under the repo checkpoint
  directory on the GPU server.
- Run a real DC-AE shape test:

```text
flow RGB [B, 3, 256, 256]
-> DC-AE encode
-> [B, 512, 2, 2]
-> tokens [B, 4, 512]
```

- Inspect the real human video data layout and decide whether to cut segments or write an episode+JSON dataset.
- Inspect local HPC data roots:

```text
/data/user/wsong890/data
```

  Look for known public human datasets, egocentric video collections, and robot
  datasets that already provide image/action pairs.
  Current check found only an empty `EgoDex` directory there.

- If needed, inspect the separate human-video server and pull one sample video
  plus one JSON only:

```text
host alias: lft-4090-2
path: /data/LFT-W02_data/shuaizhou/human_video_data/D0_huamn_dataset
```

  EgoVerse is present there with this layout:

```text
EgoVerse/annotation/<task>/<sample_id>.json
EgoVerse/video/<task>/<sample_id>.mp4
```

  Pulled one small sample into the HPC repo for loader reference:

```text
dataset/human_data/egoverse_sample/freeform_do_dishes/305.json
dataset/human_data/egoverse_sample/freeform_do_dishes/305.mp4
```

  Sample metadata:

```text
task: freeform_do_dishes
sample_id: 305
video: 640x480, 30 fps, 500 frames, 16.67 s
json: description/fps/sample_id/nframes/segments
segment: start_frame=0, end_frame=491, instruction="Wash the blue bowl with sponge"
```

- Labeled robot data format:

```text
image_t / image_t+1 / action_t
camera: observations/images/cam_high
action key: action
action dim: 14
```

- Implement mixed training batches:

```text
90% unlabeled flow reconstruction
10% labeled trajectory weak action supervision
```

- Mixed training is implemented in `train_mixed.py`. Current v0 hyperparameters
  are `action_weight=1.0` and `beta=1e-4`; Motus does not publish these values,
  so they are placeholders for the first runnable pipeline.
- Add export pipeline from trained checkpoint to Motus format:

```text
human_ego_pretrain/latent_action_dim14/*.pt
```

- Add a dataloader smoke test against `data/latent_action/latent_action_dataset.py`.

## V0 Mixed-Data Training Plan

Use this first-pass recipe for latent-action VAE training:

- Unlabeled reconstruction data: EgoVerse egocentric human videos copied under
  `dataset/human_data/egoverse_raw/EgoVerse`, excluding `debug` for the first run.
- Labeled weak action-supervision data:
  `/data/user/wsong890/lifuhao/Data/aloha_preprocessed`.
- Robot camera: `observations/images/cam_high`.
- Robot action label: real `action` with shape `[T, 14]`.
- Do not use `relative_action` in v0 because the Motus loss is written against
  `a_real` and describes alignment to the true control distribution.
- Mixed sampling: batch-level 9 unlabeled batches for every 1 labeled batch.

New scripts:

```bash
python -m latent_action_vae.build_manifests
python -m latent_action_vae.cache_flows --source both --flow-backend ptlflow --ptlflow-model dpflow --ptlflow-ckpt latent_action_vae/checkpoints/dpflow/dpflow-things-2012b5d6.ckpt
python -m latent_action_vae.train_mixed --flow-backend ptlflow --ptlflow-model dpflow --ptlflow-ckpt latent_action_vae/checkpoints/dpflow/dpflow-things-2012b5d6.ckpt
```

The flow backend is DPFlow through PTLFlow. The current HPC copy stores the
official `things` checkpoint at:

```text
latent_action_vae/checkpoints/dpflow/dpflow-things-2012b5d6.ckpt
```

OpenCV flow is intentionally not used for this pipeline.

## HPC GPU Test Command

Use an interactive debug allocation before running GPU-only checks:

```bash
srun -p debug -N 1 -n 1 --cpus-per-task=8 --gres=gpu:1 --pty bash
conda activate /data/user/wsong890/envs/shuai_d0
cd /data/user/wsong890/shuaizhou/d0
python -m latent_action_vae.smoke_test
```
