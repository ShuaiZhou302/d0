from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
from torch.nn import functional as F


def load_flow_tensor(path: str | Path, key: str = "flow") -> torch.Tensor:
    """Load a flow tensor as [N, 2, H, W]."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        if key in data:
            data = data[key]
        elif "optical_flow" in data:
            data = data["optical_flow"]
        elif "flow" in data:
            data = data["flow"]
        else:
            raise KeyError(f"{path} has no '{key}', 'flow', or 'optical_flow' tensor")
    if not isinstance(data, torch.Tensor):
        raise TypeError(f"Expected Tensor or dict of Tensor at {path}, got {type(data)}")
    flow = data.float()
    if flow.dim() == 3:
        flow = flow.unsqueeze(0)
    if flow.dim() != 4:
        raise ValueError(f"Expected [N,2,H,W] or [N,H,W,2], got {tuple(flow.shape)} at {path}")
    if flow.shape[1] != 2 and flow.shape[-1] == 2:
        flow = flow.permute(0, 3, 1, 2).contiguous()
    if flow.shape[1] != 2:
        raise ValueError(f"Expected flow channel dim of 2, got {tuple(flow.shape)} at {path}")
    return flow


def iter_flow_files(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    files = sorted(root.rglob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt flow files found under {root}")
    return files


def resize_flow(flow: torch.Tensor, size: int) -> torch.Tensor:
    if flow.shape[-2:] == (size, size):
        return flow
    return F.interpolate(flow, size=(size, size), mode="bilinear", align_corners=False)


def flow_xy_to_rgb(flow: torch.Tensor, max_flow: float) -> torch.Tensor:
    """Convert XY optical flow to a deterministic HSV-style RGB image.

    Args:
        flow: Tensor with shape [N, 2, H, W].
        max_flow: Flow magnitude mapped to full value/saturation before clipping.

    Returns:
        Tensor with shape [N, 3, H, W] in [0, 1].
    """
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError(f"Expected [N,2,H,W] flow, got {tuple(flow.shape)}")
    u = flow[:, 0]
    v = flow[:, 1]
    angle = torch.atan2(v, u)
    hue = (angle + torch.pi) / (2.0 * torch.pi)
    magnitude = torch.sqrt(u.square() + v.square())
    value = (magnitude / float(max_flow)).clamp(0.0, 1.0)
    saturation = torch.ones_like(value)
    return hsv_to_rgb(hue, saturation, value)


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Vectorized HSV to RGB for tensors shaped [N, H, W], output [N, 3, H, W]."""
    h6 = (h * 6.0) % 6.0
    i = torch.floor(h6).long()
    f = h6 - i.float()
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    r = torch.empty_like(h)
    g = torch.empty_like(h)
    b = torch.empty_like(h)
    masks = [i == k for k in range(6)]

    r[masks[0]], g[masks[0]], b[masks[0]] = v[masks[0]], t[masks[0]], p[masks[0]]
    r[masks[1]], g[masks[1]], b[masks[1]] = q[masks[1]], v[masks[1]], p[masks[1]]
    r[masks[2]], g[masks[2]], b[masks[2]] = p[masks[2]], v[masks[2]], t[masks[2]]
    r[masks[3]], g[masks[3]], b[masks[3]] = p[masks[3]], q[masks[3]], v[masks[3]]
    r[masks[4]], g[masks[4]], b[masks[4]] = t[masks[4]], p[masks[4]], v[masks[4]]
    r[masks[5]], g[masks[5]], b[masks[5]] = v[masks[5]], p[masks[5]], q[masks[5]]
    return torch.stack([r, g, b], dim=1).clamp(0.0, 1.0)


def normalize_flow_rgb(flow_rgb: torch.Tensor) -> torch.Tensor:
    return flow_rgb.mul(2.0).sub(1.0)


def denormalize_flow_rgb(flow_rgb: torch.Tensor) -> torch.Tensor:
    return flow_rgb.add(1.0).mul(0.5).clamp(0.0, 1.0)


def normalize_flow(flow: torch.Tensor, max_flow: float) -> torch.Tensor:
    return (flow / float(max_flow)).clamp(-1.0, 1.0)


def denormalize_flow(flow: torch.Tensor, max_flow: float) -> torch.Tensor:
    return flow * float(max_flow)


def save_latent_actions(latents: torch.Tensor, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(latents.detach().cpu().float(), output_path)


def stack_limited_flow_rgbs(files: Iterable[Path], max_items: int | None, input_size: int, max_flow: float) -> torch.Tensor:
    chunks = []
    total = 0
    for path in files:
        flow = resize_flow(load_flow_tensor(path), input_size)
        flow = normalize_flow_rgb(flow_xy_to_rgb(flow, max_flow))
        if max_items is not None and total + flow.shape[0] > max_items:
            flow = flow[: max_items - total]
        chunks.append(flow)
        total += flow.shape[0]
        if max_items is not None and total >= max_items:
            break
    if not chunks:
        raise RuntimeError("No flow frames loaded")
    return torch.cat(chunks, dim=0)


def make_synthetic_flows(num_frames: int = 128, size: int = 64) -> torch.Tensor:
    """Create simple smooth translation-like flow fields for smoke tests."""
    y = torch.linspace(-1.0, 1.0, size)
    x = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    flows = []
    for i in range(num_frames):
        angle = 2.0 * torch.pi * torch.tensor(float(i) / max(num_frames, 1))
        amp = 0.35 + 0.15 * torch.sin(angle * 3.0)
        blob = torch.exp(-((xx * 1.8) ** 2 + (yy * 1.4) ** 2))
        swirl_x = -yy * 0.15 * torch.cos(angle)
        swirl_y = xx * 0.15 * torch.sin(angle)
        flow_x = amp * torch.cos(angle) * blob + swirl_x
        flow_y = amp * torch.sin(angle) * blob + swirl_y
        flows.append(torch.stack([flow_x, flow_y], dim=0))
    return torch.stack(flows, dim=0)
