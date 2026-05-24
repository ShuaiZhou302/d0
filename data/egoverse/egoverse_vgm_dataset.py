from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.utils.data as data
from transformers import AutoProcessor

from data.utils.image_utils import load_video_frames, tensor_to_pil
from utils.vlm_utils import preprocess_vlm_messages


logger = logging.getLogger(__name__)


class EgoVerseTrimodalDataset(data.Dataset):
    """Segment-level EgoVerse dataset for human-video trimodal training.

    The first smoke path uses zero action tokens so Motus still runs through
    Video + Action + Understanding joint attention while action loss is disabled.
    Later, these zero tokens can be replaced by latent actions exported by the
    latent action VAE.
    """

    def __init__(
        self,
        *,
        train_manifest: str | None = None,
        val_manifest: str | None = None,
        manifest: str | None = None,
        global_downsample_rate: int = 2,
        video_action_freq_ratio: int = 1,
        num_video_frames: int = 8,
        action_dim: int = 14,
        action_mode: str = "zeros",
        video_size: Tuple[int, int] = (384, 320),
        image_aug: bool = False,
        vlm_checkpoint_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        val: bool = False,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.manifest = Path(manifest or (val_manifest if val else train_manifest))
        self.global_downsample_rate = int(global_downsample_rate)
        self.video_action_freq_ratio = int(video_action_freq_ratio)
        self.num_video_frames = int(num_video_frames)
        self.action_dim = int(action_dim)
        self.action_chunk_size = self.num_video_frames * self.video_action_freq_ratio
        self.action_mode = str(action_mode)
        if self.action_mode not in {"none", "zeros"}:
            raise ValueError(f"Unsupported action_mode={self.action_mode}; expected 'none' or 'zeros'")
        self.video_size = video_size
        self.image_aug = bool(image_aug and not val)
        self.val = bool(val)
        self.seed = int(seed)
        self.rows = self._load_rows(self.manifest)
        if max_samples is not None and int(max_samples) > 0:
            self.rows = self.rows[: int(max_samples)]

        self.vlm_processor = None
        if vlm_checkpoint_path:
            try:
                self.vlm_processor = AutoProcessor.from_pretrained(vlm_checkpoint_path)
                logger.info("Loaded VLM processor from %s", vlm_checkpoint_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load VLM processor from %s: %s", vlm_checkpoint_path, exc)

        logger.info(
            "EgoVerseTrimodalDataset initialized: manifest=%s samples=%d val=%s action_mode=%s action_chunk_size=%d",
            self.manifest,
            len(self.rows),
            self.val,
            self.action_mode,
            self.action_chunk_size,
        )

    @staticmethod
    def _load_rows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"EgoVerse trimodal manifest not found: {path}")
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            raise ValueError(f"EgoVerse trimodal manifest is empty: {path}")
        return rows

    def __len__(self) -> int:
        return len(self.rows) * (1 if self.val else 100)

    def _select_indices(self, row: dict[str, Any], idx: int) -> tuple[int, list[int]]:
        step = self.global_downsample_rate
        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        max_condition = end_frame - self.num_video_frames * step
        if max_condition <= start_frame:
            condition_idx = start_frame
        elif self.val:
            condition_idx = (start_frame + max_condition) // 2
        else:
            rng = random.Random(self.seed + idx)
            condition_idx = rng.randint(start_frame, max_condition)
        video_indices = [condition_idx + (i + 1) * step for i in range(self.num_video_frames)]
        video_indices = [min(frame_idx, end_frame - 1) for frame_idx in video_indices]
        return condition_idx, video_indices

    @staticmethod
    def _load_language_embedding(path: str) -> tuple[torch.Tensor, int]:
        data_obj = torch.load(path, map_location="cpu")
        if isinstance(data_obj, list):
            embedding = data_obj[0]
            selected_idx = 0
        elif isinstance(data_obj, torch.Tensor):
            embedding = data_obj
            selected_idx = 0
        else:
            raise TypeError(f"Unsupported UMT5 embedding type at {path}: {type(data_obj)}")
        if embedding.dim() == 3 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        if not isinstance(embedding, torch.Tensor) or embedding.dim() != 2:
            raise ValueError(f"Expected UMT5 embedding [S,D], got {type(embedding)} {getattr(embedding, 'shape', None)} at {path}")
        return embedding.float(), selected_idx

    def __getitem__(self, idx: int) -> Optional[dict[str, Any]]:
        if not self.rows:
            return None
        row = self.rows[idx % len(self.rows)]
        try:
            condition_idx, video_indices = self._select_indices(row, idx)
            frame_indices = [condition_idx] + video_indices
            frames = load_video_frames(row["video_path"], frame_indices, self.video_size)
            first_frame = frames[0]
            video_frames = frames[1:]
            language_embedding, _ = self._load_language_embedding(row["umt5_path"])

            vlm_inputs = None
            if self.vlm_processor is not None:
                first_frame_pil = tensor_to_pil(first_frame)
                vlm_inputs = preprocess_vlm_messages(row["instruction"], first_frame_pil, self.vlm_processor)

            sample = {
                "first_frame": first_frame,
                "video_frames": video_frames,
                "language_embedding": language_embedding,
                "vlm_inputs": vlm_inputs,
                "dataset_name": "egoverse_trimodal",
                "sample_id": row["id"],
            }
            if self.action_mode == "zeros":
                action_sequence = torch.zeros(self.action_chunk_size, self.action_dim, dtype=torch.float32)
                initial_state = torch.zeros(self.action_dim, dtype=torch.float32)
                sample["initial_state"] = initial_state
                sample["action_sequence"] = action_sequence
                sample["action_mask"] = torch.ones_like(action_sequence, dtype=torch.bool)
            return sample
        except Exception as exc:  # noqa: BLE001
            logger.error("Error loading EgoVerse sample %s: %s", row.get("id", "unknown"), exc)
            return None
