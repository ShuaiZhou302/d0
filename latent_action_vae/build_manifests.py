from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from latent_action_vae.datasets import write_jsonl


DEFAULT_CAMERA_KEY = "observations/images/cam_high"
DEFAULT_ACTION_KEY = "action"


def build_egoverse_manifest(root: Path, output: Path, *, include_debug: bool = False) -> int:
    annotation_root = root / "EgoVerse" / "annotation"
    video_root = root / "EgoVerse" / "video"
    rows = []
    for json_path in sorted(annotation_root.glob("*/*.json")):
        task = json_path.parent.name
        if task == "debug" and not include_debug:
            continue
        sample_id = json_path.stem
        video_path = video_root / task / f"{sample_id}.mp4"
        if not video_path.exists():
            continue
        with json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        rows.append(
            {
                "kind": "egoverse",
                "task": task,
                "sample_id": str(meta.get("sample_id", sample_id)),
                "video_path": str(video_path),
                "json_path": str(json_path),
                "fps": int(meta.get("fps", 0)),
                "nframes": int(meta.get("nframes", 0)),
                "description": meta.get("description", ""),
                "segments": meta.get("segments", []),
            }
        )
    write_jsonl(rows, output)
    return len(rows)


def build_aloha_manifest(root: Path, output: Path, *, camera_key: str, action_key: str) -> tuple[int, int]:
    rows = []
    total_steps = 0
    for hdf5_path in sorted(root.glob("*/*/*.hdf5")):
        task = hdf5_path.parents[1].name
        split = hdf5_path.parent.name
        episode = hdf5_path.stem
        with h5py.File(hdf5_path, "r") as f:
            if camera_key not in f:
                raise KeyError(f"{hdf5_path} missing camera_key={camera_key}")
            if action_key not in f:
                raise KeyError(f"{hdf5_path} missing action_key={action_key}")
            action_shape = tuple(f[action_key].shape)
            if len(action_shape) != 2 or action_shape[1] != 14:
                raise ValueError(f"{hdf5_path} expected action shape [T,14], got {action_shape}")
            image_shape = tuple(f[camera_key].shape)
            if len(image_shape) != 4 or image_shape[-1] != 3:
                raise ValueError(f"{hdf5_path} expected camera image shape [T,H,W,3], got {image_shape}")
            length = min(int(action_shape[0]), int(image_shape[0]))
        total_steps += length
        rows.append(
            {
                "kind": "aloha",
                "task": task,
                "split": split,
                "episode": episode,
                "hdf5_path": str(hdf5_path),
                "length": length,
                "camera_key": camera_key,
                "action_key": action_key,
                "action_dim": 14,
            }
        )
    write_jsonl(rows, output)
    return len(rows), total_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EgoVerse and ALOHA manifests for latent action VAE training.")
    parser.add_argument("--egoverse-root", type=Path, default=Path("dataset/human_data/egoverse_raw"))
    parser.add_argument("--aloha-root", type=Path, default=Path("/data/user/wsong890/lifuhao/Data/aloha_preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/human_data/manifests"))
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--camera-key", default=DEFAULT_CAMERA_KEY)
    parser.add_argument("--action-key", default=DEFAULT_ACTION_KEY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ego_count = build_egoverse_manifest(
        args.egoverse_root,
        args.output_dir / "egoverse_unlabeled.jsonl",
        include_debug=args.include_debug,
    )
    aloha_count, aloha_steps = build_aloha_manifest(
        args.aloha_root,
        args.output_dir / "aloha_labeled.jsonl",
        camera_key=args.camera_key,
        action_key=args.action_key,
    )
    print(f"egoverse_entries={ego_count}")
    print(f"aloha_episodes={aloha_count}")
    print(f"aloha_action_steps={aloha_steps}")
    print(f"camera_key={args.camera_key}")
    print(f"action_key={args.action_key}")


if __name__ == "__main__":
    main()
