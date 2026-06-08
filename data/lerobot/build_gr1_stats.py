#!/usr/bin/env python3
"""Build Motus normalization stats for GR1 LeRobot task folders.

The GR1 dataset ships per-task `meta/stats.json` files. Motus expects a single
JSON object keyed by embodiment name with direct `min`/`max` arrays. This script
combines per-task action min/max by taking dimension-wise global min/max.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def iter_task_dirs(root: Path, tasks: list[str] | None) -> Iterable[Path]:
    if tasks:
        for task in tasks:
            path = root / task
            if not path.is_dir():
                raise FileNotFoundError(f"Task directory not found: {path}")
            yield path
        return

    yield from sorted(p for p in root.glob("gr1_arms_waist.*") if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Root containing gr1_arms_waist.* task dirs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-key", default="gr1_arms_waist")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task dirs to include")
    args = parser.parse_args()

    mins = []
    maxs = []
    used_tasks = []
    for task_dir in iter_task_dirs(args.root, args.tasks):
        stats_path = task_dir / "meta" / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Missing stats: {stats_path}")
        stats = json.loads(stats_path.read_text())
        action_stats = stats["action"]
        mins.append(np.asarray(action_stats["min"], dtype=np.float32))
        maxs.append(np.asarray(action_stats["max"], dtype=np.float32))
        used_tasks.append(task_dir.name)

    if not mins:
        raise ValueError(f"No GR1 task dirs found under {args.root}")

    action_min = np.stack(mins, axis=0).min(axis=0)
    action_max = np.stack(maxs, axis=0).max(axis=0)
    if action_min.shape != action_max.shape:
        raise ValueError(f"min/max shape mismatch: {action_min.shape} vs {action_max.shape}")

    payload = {
        args.stats_key: {
            "min": action_min.tolist(),
            "max": action_max.tolist(),
            "action_dim": int(action_min.shape[0]),
            "source": "merged_gr1_meta_stats",
            "num_tasks": len(used_tasks),
            "tasks": used_tasks,
        }
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote={args.output}")
    print(f"stats_key={args.stats_key}")
    print(f"action_dim={action_min.shape[0]}")
    print(f"tasks={len(used_tasks)}")


if __name__ == "__main__":
    main()
