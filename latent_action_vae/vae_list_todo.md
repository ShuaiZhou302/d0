# Latent Action VAE TODO

## Design Choices

1. `a_pred = mu`, which is the encoder output 14D latent action; no extra `action_head`.

2. VAE reconstruction uses sampled `z`; action alignment and export use deterministic `mu`.

3. Lightweight encoder/decoder uses `MLP + LayerNorm + SiLU`, because the paper does not specify the exact structure.

4. DC-AE is trainable by default, not frozen; the paper does not specify freeze vs finetune.

5. Flow RGB uses HSV-style optical-flow colorization; the paper does not specify the color convention.

6. DPFlow uses official PTLFlow `dpflow`; current checkpoint is `dpflow-things`, because the paper does not specify the exact checkpoint.

7. Loss weights currently use `beta=1e-4` and `action_weight=1.0`; the paper does not specify lambda_a or beta.

## Fixed V0 Settings

1. 10% labeled data uses ALOHA `action`; do not use `relative_action`, because the paper writes real action / true control distribution.

2. Robot camera uses `observations/images/cam_high`.

3. Mixed ratio uses batch-level 9:1: EgoVerse unlabeled : ALOHA labeled.

## Current Data Status

1. EgoVerse rsync is still running on HPC.

2. Last checked partial EgoVerse copy:

```text
video: 2382
annotation: 7574
size: 116G
tmux session: motus_latent_shuai
```

3. Existing manifests are partial and must be regenerated after rsync finishes:

```text
egoverse_unlabeled.jsonl: 1810
aloha_labeled.jsonl: 787
```

## TODO

1. Wait for EgoVerse rsync to finish.

2. Regenerate EgoVerse manifest.

3. Run DPFlow cache generation.

4. Run 500-1000 step sanity training.

5. Check loss curves for NaN/OOM and whether recon/action/KL are finite.

6. Reload the checkpoint once to verify it is usable.
