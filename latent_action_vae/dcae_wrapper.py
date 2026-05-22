from __future__ import annotations

import importlib.util
import sys
import types
from typing import Callable, Optional
from pathlib import Path

import torch
from huggingface_hub import PyTorchModelHubMixin
from safetensors.torch import load_file
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


def install_optional_dependency_shims() -> None:
    """Keep EfficientViT importable when unused export-only deps are absent."""
    if "onnxsim" not in sys.modules:
        onnxsim = types.ModuleType("onnxsim")

        def simplify(*_args, **_kwargs):
            raise ImportError("onnxsim is required only for EfficientViT ONNX export.")

        onnxsim.simplify = simplify
        sys.modules["onnxsim"] = onnxsim


def load_dc_ae_module():
    """Load EfficientViT's DC-AE module without importing unrelated SAM/seg code."""
    add_efficientvit_to_path()
    install_optional_dependency_shims()
    module_name = "_efficientvit_dc_ae_direct"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = efficientvit_root() / "efficientvit" / "models" / "efficientvit" / "dc_ae.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load EfficientViT DC-AE module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def create_dcae_hf_class():
    dc_ae = load_dc_ae_module()
    registered: dict[str, tuple[Callable, Optional[str]]] = {
        "dc-ae-f32c32-in-1.0": (dc_ae.dc_ae_f32c32, None),
        "dc-ae-f64c128-in-1.0": (dc_ae.dc_ae_f64c128, None),
        "dc-ae-f128c512-in-1.0": (dc_ae.dc_ae_f128c512, None),
        "dc-ae-f32c32-mix-1.0": (dc_ae.dc_ae_f32c32, None),
        "dc-ae-f64c128-mix-1.0": (dc_ae.dc_ae_f64c128, None),
        "dc-ae-f128c512-mix-1.0": (dc_ae.dc_ae_f128c512, None),
        "dc-ae-f32c32-sana-1.0": (dc_ae.dc_ae_f32c32, None),
    }

    def create_model_cfg(name: str, pretrained_path: Optional[str] = None):
        if name not in registered:
            raise ValueError(f"{name} is not a supported DC-AE model")
        dc_ae_cls, default_pt_path = registered[name]
        return dc_ae_cls(name, default_pt_path if pretrained_path is None else pretrained_path)

    class DCAE_HF(dc_ae.DCAE, PyTorchModelHubMixin):
        def __init__(self, model_name: str):
            super().__init__(create_model_cfg(model_name))

    return DCAE_HF


def load_dcae_from_local_dir(model_dir: Path, *, device: torch.device | str, dtype: torch.dtype) -> nn.Module:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected config.json and model.safetensors in local DC-AE directory: {model_dir}"
        )

    import json

    model_name = json.loads(config_path.read_text())["model_name"]
    DCAE_HF = create_dcae_hf_class()
    dcae = DCAE_HF(model_name)
    dcae.load_state_dict(load_file(str(weights_path), device="cpu"))
    return dcae.to(device=device, dtype=dtype)


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
    install_optional_dependency_shims()

    model_path = Path(model_name).expanduser()
    if model_path.exists():
        dcae = load_dcae_from_local_dir(model_path, device=device, dtype=dtype)
    else:
        DCAE_HF = create_dcae_hf_class()
        if "/" not in model_name:
            model_name = f"mit-han-lab/{model_name}"
        dcae = DCAE_HF.from_pretrained(model_name).to(device=device, dtype=dtype)

    dcae.train(mode=trainable)
    for param in dcae.parameters():
        param.requires_grad_(trainable)
    return dcae
