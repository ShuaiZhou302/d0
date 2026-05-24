#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ManifestConfig:
    raw_root: Path
    output_root: Path
    train_ratio: float
    seed: int
    num_video_frames: int
    global_downsample_rate: int
    min_confidence: float
    max_videos: int | None

    @property
    def min_required_frames(self) -> int:
        return 1 + self.num_video_frames * self.global_downsample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val manifests for EgoVerse human-video training.")
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/human_data/egoverse_raw/EgoVerse"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset/human_data/egoverse_vgm"))
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--global-downsample-rate", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-videos", type=int, default=0, help="Limit videos for smoke tests; 0 means all.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sample_key(video_path: Path) -> tuple[str, str]:
    return video_path.parent.name, video_path.stem


def collect_videos(cfg: ManifestConfig) -> list[Path]:
    video_root = cfg.raw_root / "video"
    annotation_root = cfg.raw_root / "annotation"
    videos = []
    for video_path in sorted(video_root.glob("*/*.mp4")):
        task, sample_id = sample_key(video_path)
        json_path = annotation_root / task / f"{sample_id}.json"
        if json_path.exists():
            videos.append(video_path)
    if cfg.max_videos is not None and cfg.max_videos > 0:
        videos = videos[: cfg.max_videos]
    return videos


def split_videos(videos: list[Path], train_ratio: float, seed: int) -> dict[tuple[str, str], str]:
    keys = [sample_key(path) for path in videos]
    rng = random.Random(seed)
    shuffled = list(keys)
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_ratio))
    train_keys = set(shuffled[:n_train])
    return {key: ("train" if key in train_keys else "val") for key in keys}


def safe_id(task: str, sample_id: str, seg_id: int) -> str:
    return f"{task}__{sample_id}__seg{int(seg_id):03d}"


def build_rows(cfg: ManifestConfig) -> list[dict[str, Any]]:
    videos = collect_videos(cfg)
    split_by_key = split_videos(videos, cfg.train_ratio, cfg.seed)
    annotation_root = cfg.raw_root / "annotation"
    rows: list[dict[str, Any]] = []
    for video_path in videos:
        task, sample_id = sample_key(video_path)
        split = split_by_key[(task, sample_id)]
        json_path = annotation_root / task / f"{sample_id}.json"
        meta = load_json(json_path)
        fps = int(meta.get("fps", 30))
        nframes = int(meta.get("nframes", 0))
        for segment in meta.get("segments", []):
            confidence = float(segment.get("confidence", 1.0))
            if confidence < cfg.min_confidence:
                continue
            start_frame = int(segment["start_frame"])
            end_frame = int(segment["end_frame"])
            if end_frame - start_frame < cfg.min_required_frames:
                continue
            seg_id = int(segment.get("seg_id", len(rows)))
            instruction = str(segment.get("instruction", "")).strip()
            if not instruction:
                continue
            item_id = safe_id(task, sample_id, seg_id)
            meta_path = cfg.output_root / "metas" / split / f"{item_id}.txt"
            umt5_path = cfg.output_root / "umt5_wan" / split / f"{item_id}.pt"
            rows.append(
                {
                    "id": item_id,
                    "split": split,
                    "task": task,
                    "sample_id": sample_id,
                    "seg_id": seg_id,
                    "video_path": str(video_path),
                    "json_path": str(json_path),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": fps,
                    "nframes": nframes,
                    "instruction": instruction,
                    "meta_path": str(meta_path),
                    "umt5_path": str(umt5_path),
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metas(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        meta_path = Path(row["meta_path"])
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(str(row["instruction"]).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = ManifestConfig(
        raw_root=args.raw_root,
        output_root=args.output_root,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        num_video_frames=int(args.num_video_frames),
        global_downsample_rate=int(args.global_downsample_rate),
        min_confidence=float(args.min_confidence),
        max_videos=int(args.max_videos) if int(args.max_videos) > 0 else None,
    )
    rows = build_rows(cfg)
    train_rows = [row for row in rows if row["split"] == "train"]
    val_rows = [row for row in rows if row["split"] == "val"]
    write_jsonl(cfg.output_root / "manifests" / "train.jsonl", train_rows)
    write_jsonl(cfg.output_root / "manifests" / "val.jsonl", val_rows)
    write_metas(rows)
    train_keys = {(row["task"], row["sample_id"]) for row in train_rows}
    val_keys = {(row["task"], row["sample_id"]) for row in val_rows}
    overlap = train_keys & val_keys
    if overlap:
        raise RuntimeError(f"Train/val split leaked videos: {sorted(overlap)[:5]}")
    print(
        "built EgoVerse trimodal manifests: "
        f"train_segments={len(train_rows)} val_segments={len(val_rows)} "
        f"train_videos={len(train_keys)} val_videos={len(val_keys)} "
        f"min_required_frames={cfg.min_required_frames}"
    )


if __name__ == "__main__":
    main()
