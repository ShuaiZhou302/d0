from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


EGO_SAMPLE_KIND = "egoverse"
ALOHA_SAMPLE_KIND = "aloha"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def image_to_float_chw(image: np.ndarray, image_size: int) -> torch.Tensor:
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape={image.shape}")
    if image.shape[0] != image_size or image.shape[1] != image_size:
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()
    return tensor / 255.0


def read_video_frame_rgb(video_path: str | Path, frame_index: int, image_size: int) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture failed to open: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return image_to_float_chw(frame_rgb, image_size)


def valid_segment_ranges(entry: dict[str, Any], min_len: int = 2) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    nframes = int(entry.get("nframes", 0))
    for seg in entry.get("segments", []) or []:
        start = max(0, int(seg.get("start_frame", 0)))
        end = min(nframes, int(seg.get("end_frame", nframes)))
        if end - start >= min_len:
            ranges.append((start, end))
    if not ranges and nframes >= min_len:
        ranges.append((0, nframes))
    return ranges


@dataclass(frozen=True)
class FramePair:
    sample_id: str
    source: str
    frame_index: int
    frame0: torch.Tensor
    frame1: torch.Tensor
    action: torch.Tensor | None = None


class EgoVerseUnlabeledDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int = 256,
        samples_per_video: int = 8,
        seed: int = 0,
    ) -> None:
        self.entries = read_jsonl(manifest_path)
        self.image_size = int(image_size)
        self.samples: list[tuple[int, int]] = []
        rng = random.Random(seed)
        for entry_idx, entry in enumerate(self.entries):
            pairs: list[int] = []
            for start, end in valid_segment_ranges(entry, min_len=2):
                pairs.extend(range(start, end - 1))
            if not pairs:
                continue
            if samples_per_video > 0 and len(pairs) > samples_per_video:
                pairs = sorted(rng.sample(pairs, samples_per_video))
            for frame_idx in pairs:
                self.samples.append((entry_idx, frame_idx))
        if not self.samples:
            raise RuntimeError(f"No frame pairs found in EgoVerse manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FramePair:
        entry_idx, frame_idx = self.samples[idx]
        entry = self.entries[entry_idx]
        frame0 = read_video_frame_rgb(entry["video_path"], frame_idx, self.image_size)
        frame1 = read_video_frame_rgb(entry["video_path"], frame_idx + 1, self.image_size)
        sample_id = f"{entry['task']}/{entry['sample_id']}/{frame_idx:06d}"
        return FramePair(sample_id=sample_id, source=EGO_SAMPLE_KIND, frame_index=frame_idx, frame0=frame0, frame1=frame1)


class AlohaLabeledDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int = 256,
        samples_per_episode: int = 8,
        seed: int = 0,
    ) -> None:
        self.entries = read_jsonl(manifest_path)
        self.image_size = int(image_size)
        self.samples: list[tuple[int, int]] = []
        rng = random.Random(seed)
        for entry_idx, entry in enumerate(self.entries):
            length = int(entry["length"])
            pairs = list(range(max(0, length - 1)))
            if not pairs:
                continue
            if samples_per_episode > 0 and len(pairs) > samples_per_episode:
                pairs = sorted(rng.sample(pairs, samples_per_episode))
            for frame_idx in pairs:
                self.samples.append((entry_idx, frame_idx))
        if not self.samples:
            raise RuntimeError(f"No frame pairs found in ALOHA manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FramePair:
        entry_idx, frame_idx = self.samples[idx]
        entry = self.entries[entry_idx]
        with h5py.File(entry["hdf5_path"], "r") as f:
            images = f[entry["camera_key"]]
            action = torch.from_numpy(np.asarray(f[entry["action_key"]][frame_idx])).float()
            frame0 = image_to_float_chw(np.asarray(images[frame_idx]), self.image_size)
            frame1 = image_to_float_chw(np.asarray(images[frame_idx + 1]), self.image_size)
        sample_id = f"{entry['task']}/{entry['split']}/{entry['episode']}/{frame_idx:06d}"
        return FramePair(
            sample_id=sample_id,
            source=ALOHA_SAMPLE_KIND,
            frame_index=frame_idx,
            frame0=frame0,
            frame1=frame1,
            action=action,
        )


def collate_frame_pairs(batch: list[FramePair]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_id": [item.sample_id for item in batch],
        "source": [item.source for item in batch],
        "frame_index": torch.tensor([item.frame_index for item in batch], dtype=torch.long),
        "frame0": torch.stack([item.frame0 for item in batch], dim=0),
        "frame1": torch.stack([item.frame1 for item in batch], dim=0),
    }
    if batch[0].action is not None:
        result["action"] = torch.stack([item.action for item in batch if item.action is not None], dim=0)
    return result
