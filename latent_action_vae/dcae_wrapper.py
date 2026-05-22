from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


DEFAULT_DCAE_MODEL = "dc-ae-f128c512-mix-1.0"


def efficientvit_root() -> Path:
    return Path(__file__).resolve().parent / "third_party" / "efficientvit"


def add_efficientvit_to_path() -> None:
    root = efficientvit_root()
    if not root.exists():
        raise FileNotFoundError(
            f"EfficientViT submodule not found at {root}. "
            "Run: git submodule update --init --recursive latent_action_vae/third_party/efficientvit"
        )
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def load_dcae(
    model_name: str = DEFAULT_DCAE_MODEL,
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    trainable: bool = True,
) -> nn.Module:
    """Load the public DC-AE checkpoint through EfficientViT/Hugging Face.

    The returned module is trainable by default because the Motus latent-action
    VAE training objective reconstructs flow RGB, not natural images.
    """
    add_efficientvit_to_path()
    from efficientvit.ae_model_zoo import DCAE_HF

    if "/" not in model_name:
        model_name = f"mit-han-lab/{model_name}"
    dcae = DCAE_HF.from_pretrained(model_name).to(device=device, dtype=dtype)
    dcae.train(mode=trainable)
    for param in dcae.parameters():
        param.requires_grad_(trainable)
    return dcae
