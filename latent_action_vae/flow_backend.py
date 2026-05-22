from __future__ import annotations

import os
import sys
import types
import importlib
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch


class FlowBackend(Protocol):
    def compute(self, frame0: torch.Tensor, frame1: torch.Tensor) -> torch.Tensor:
        """Return optical flow as [2,H,W] float tensor in pixel units."""


class OpenCVFarnebackBackend:
    """CPU fallback for smoke tests when the paper DPFlow backend is unavailable."""

    name = "opencv_farneback"

    def compute(self, frame0: torch.Tensor, frame1: torch.Tensor) -> torch.Tensor:
        image0 = tensor_chw_to_uint8_rgb(frame0)
        image1 = tensor_chw_to_uint8_rgb(frame1)
        gray0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray0,
            gray1,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        return torch.from_numpy(flow).permute(2, 0, 1).float()


class PTLFlowBackend:
    """PTLFlow wrapper. Use `model_name=dpflow` if that model is installed."""

    def __init__(self, model_name: str = "dpflow", ckpt_path: str | None = None, device: str = "cuda") -> None:
        third_party = Path(__file__).resolve().parent / "third_party" / "ptlflow"
        if third_party.exists() and str(third_party) not in sys.path:
            sys.path.insert(0, str(third_party))
        if model_name == "dpflow":
            os.environ.setdefault("PTLFLOW_MODEL_IMPORTS", "dpflow")
            install_dpflow_only_ptlflow_models_package(third_party)
        from ptlflow import get_model, get_model_names
        if model_name == "dpflow":
            importlib.import_module("ptlflow.models.dpflow")

        available = get_model_names()
        if model_name not in available:
            raise ValueError(
                f"PTLFlow model '{model_name}' is not registered. "
                f"Available examples: {', '.join(available[:30])}"
            )
        self.name = f"ptlflow_{model_name}"
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = get_model(model_name, ckpt_path=ckpt_path).to(self.device).eval()

    @torch.no_grad()
    def compute(self, frame0: torch.Tensor, frame1: torch.Tensor) -> torch.Tensor:
        images = torch.stack([frame0, frame1], dim=0).unsqueeze(0).to(self.device)
        images = images.float() * 255.0
        outputs = self.model({"images": images})
        flow = outputs["flows"]
        if flow.dim() == 5:
            flow = flow[:, 0]
        return flow[0].detach().cpu().float()


def tensor_chw_to_uint8_rgb(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.ascontiguousarray((array * 255.0).round().astype(np.uint8))


def build_flow_backend(
    backend: str,
    *,
    model_name: str = "dpflow",
    ckpt_path: str | None = None,
    device: str = "cuda",
) -> FlowBackend:
    if backend == "opencv":
        return OpenCVFarnebackBackend()
    if backend == "ptlflow":
        return PTLFlowBackend(model_name=model_name, ckpt_path=ckpt_path, device=device)
    raise ValueError(f"Unknown flow backend: {backend}")


def install_dpflow_only_ptlflow_models_package(third_party: Path) -> None:
    """Avoid importing every PTLFlow model when only DPFlow is needed."""
    if "ptlflow.models" in sys.modules:
        return
    models_pkg = types.ModuleType("ptlflow.models")
    models_pkg.__path__ = [str(third_party / "ptlflow" / "models")]
    sys.modules["ptlflow.models"] = models_pkg
