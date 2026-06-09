#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = (
    "ego4d_cooking_and_cleaning",
    "ego4d_other",
    "epic",
    "egoexo4d",
    "ssv2",
)


@dataclass(frozen=True)
class ManifestConfig:
    vitra_root: Path
    output_root: Path
    raw_video_root: Path | None
    train_ratio: float
    seed: int
    num_video_frames: int
    max_downsample_rate: int
    min_downsample_rate: int
    max_episodes: int | None
    prefer_rephrase: bool

    @property
    def min_required_frames(self) -> int:
        return 1 + self.num_video_frames * self.min_downsample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an EgoVerse-style manifest from extracted VITRA-1M metadata. "
            "VITRA-1M contains episode metadata, language, frame indices, and hand "
            "reconstruction; raw videos must be provided separately if training "
            "needs RGB frames."
        )
    )
    parser.add_argument("--vitra-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("dataset/human_data/vitra_vgm"))
    parser.add_argument(
        "--raw-video-root",
        type=Path,
        default=None,
        help="Optional root containing raw Ego4D/EPIC/EgoExo4D/SSV2 videos.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-video-frames", type=int, default=8)
    parser.add_argument("--max-downsample-rate", type=int, default=8)
    parser.add_argument("--min-downsample-rate", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0, help="Smoke-test limit; 0 means all.")
    parser.add_argument("--no-rephrase", action="store_true", help="Use text instead of text_rephrase.")
    return parser.parse_args()


def load_episode(path: Path) -> dict[str, Any]:
    return np.load(path, allow_pickle=True).item()


def iter_episode_paths(vitra_root: Path) -> list[Path]:
    paths: list[Path] = []
    for dataset in DATASETS:
        ann_dir = vitra_root / dataset / "episodic_annotations"
        if ann_dir.exists():
            paths.extend(sorted(ann_dir.glob("*.npy")))
    return paths


def split_episodes(paths: list[Path], train_ratio: float, seed: int) -> dict[Path, str]:
    rng = random.Random(seed)
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_train = int(round(len(shuffled) * train_ratio))
    train_paths = set(shuffled[:n_train])
    return {path: ("train" if path in train_paths else "val") for path in paths}


def safe_id(path: Path, hand: str, seg_idx: int) -> str:
    stem = path.stem.replace("/", "_")
    return f"{stem}__{hand}__seg{seg_idx:03d}"


def choose_instruction(episode: dict[str, Any], hand: str, prefer_rephrase: bool) -> list[tuple[str, tuple[int, int]]]:
    source_key = "text_rephrase" if prefer_rephrase else "text"
    source = episode.get(source_key, {}) or {}
    items = source.get(hand, []) or []
    normalized: list[tuple[str, tuple[int, int]]] = []
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        text, span = item
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        instruction = str(text).strip()
        if instruction:
            normalized.append((instruction, (int(span[0]), int(span[1]))))
    return normalized


def find_video_path(raw_video_root: Path | None, dataset: str, video_name: str) -> str | None:
    if raw_video_root is None or not video_name:
        return None

    candidates = [
        raw_video_root / dataset / video_name,
        raw_video_root / dataset / f"{video_name}.mp4",
        raw_video_root / video_name,
        raw_video_root / f"{video_name}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    # Last-resort lookup is intentionally shallow enough for smoke runs.
    matches = list(raw_video_root.glob(f"**/{video_name}")) + list(raw_video_root.glob(f"**/{video_name}.mp4"))
    if matches:
        return str(matches[0])
    return None


def build_rows(cfg: ManifestConfig) -> list[dict[str, Any]]:
    episode_paths = iter_episode_paths(cfg.vitra_root)
    if cfg.max_episodes is not None and cfg.max_episodes > 0:
        episode_paths = episode_paths[: cfg.max_episodes]
    split_by_path = split_episodes(episode_paths, cfg.train_ratio, cfg.seed)

    rows: list[dict[str, Any]] = []
    for episode_path in episode_paths:
        dataset = episode_path.parents[1].name
        split = split_by_path[episode_path]
        try:
            episode = load_episode(episode_path)
        except Exception as exc:
            rows.append(
                {
                    "id": episode_path.stem,
                    "split": split,
                    "dataset": dataset,
                    "episode_path": str(episode_path),
                    "load_error": repr(exc),
                }
            )
            continue

        video_name = str(episode.get("video_name", "")).strip()
        frame_indices = [int(x) for x in episode.get("video_decode_frame", [])]
        nframes = len(frame_indices)
        video_path = find_video_path(cfg.raw_video_root, dataset, video_name)

        for hand in ("left", "right"):
            for seg_idx, (instruction, (start, end)) in enumerate(
                choose_instruction(episode, hand, cfg.prefer_rephrase)
            ):
                start = max(0, start)
                end = min(nframes, end)
                if end - start < cfg.min_required_frames:
                    continue

                item_id = safe_id(episode_path, hand, seg_idx)
                meta_path = cfg.output_root / "metas" / split / f"{item_id}.txt"
                umt5_path = cfg.output_root / "umt5_wan" / split / f"{item_id}.pt"
                rows.append(
                    {
                        "id": item_id,
                        "split": split,
                        "dataset": dataset,
                        "sample_id": episode_path.stem,
                        "hand": hand,
                        "seg_id": seg_idx,
                        "episode_path": str(episode_path),
                        "video_name": video_name,
                        "video_path": video_path,
                        "missing_video": video_path is None,
                        "start_frame": start,
                        "end_frame": end,
                        "decode_frames": frame_indices[start:end],
                        "fps": None,
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
        if "instruction" not in row:
            continue
        meta_path = Path(row["meta_path"])
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(str(row["instruction"]).strip() + "\n", encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if "instruction" in row]
    by_split = {}
    by_dataset = {}
    missing_video = 0
    for row in valid:
        by_split[row["split"]] = by_split.get(row["split"], 0) + 1
        by_dataset[row["dataset"]] = by_dataset.get(row["dataset"], 0) + 1
        missing_video += int(bool(row.get("missing_video")))
    return {
        "rows": len(rows),
        "valid_segments": len(valid),
        "by_split": by_split,
        "by_dataset": by_dataset,
        "missing_video": missing_video,
    }


def main() -> None:
    args = parse_args()
    cfg = ManifestConfig(
        vitra_root=args.vitra_root,
        output_root=args.output_root,
        raw_video_root=args.raw_video_root,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        num_video_frames=int(args.num_video_frames),
        max_downsample_rate=int(args.max_downsample_rate),
        min_downsample_rate=int(args.min_downsample_rate),
        max_episodes=int(args.max_episodes) if int(args.max_episodes) > 0 else None,
        prefer_rephrase=not bool(args.no_rephrase),
    )
    rows = build_rows(cfg)
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    write_jsonl(cfg.output_root / "manifests" / "train.jsonl", train_rows)
    write_jsonl(cfg.output_root / "manifests" / "val.jsonl", val_rows)
    write_metas(rows)
    summary = summarize(rows)
    (cfg.output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (cfg.output_root / "manifests" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
