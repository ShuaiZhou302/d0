#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm import tqdm

from data.robotwin2.robotwin_data_convert.robotwin_converter import T5EmbeddingProcessor, process_t5_batch


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate WAN UMT5 embeddings for EgoVerse VGM manifests.")
    parser.add_argument("--manifest", type=Path, action="append", required=True, help="Manifest jsonl. Repeat for train/val.")
    parser.add_argument("--wan-repo-path", required=True, help="Directory containing models_t5_umt5-xxl-enc-bf16.pth and google/umt5-xxl.")
    parser.add_argument("--cuda-devices", default="0", help="Comma-separated CUDA device ids, e.g. 0,1,2,3.")
    parser.add_argument("--t5-max-length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-items", type=int, default=0, help="Smoke limit across all manifests; 0 means all.")
    return parser.parse_args()


def iter_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def collect_pairs(manifests: list[Path], overwrite: bool, max_items: int) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for manifest in manifests:
        for row in iter_manifest(manifest):
            meta_path = Path(row["meta_path"])
            umt5_path = Path(row["umt5_path"])
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing meta file for {row['id']}: {meta_path}")
            if umt5_path.exists() and not overwrite:
                continue
            pairs.append((str(meta_path), str(umt5_path)))
            if max_items > 0 and len(pairs) >= max_items:
                return pairs
    return pairs


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    devices = [item.strip() for item in str(args.cuda_devices).split(",") if item.strip()]
    if not devices:
        raise ValueError("--cuda-devices must contain at least one device id")
    pairs = collect_pairs(args.manifest, overwrite=bool(args.overwrite), max_items=int(args.max_items))
    logger.info("EgoVerse UMT5 pending pairs: %d", len(pairs))
    if not pairs:
        return

    chunks = [pairs[i:: len(devices)] for i in range(len(devices))]
    processors_and_chunks = []
    for device_id, chunk in zip(devices, chunks):
        processor = T5EmbeddingProcessor(
            wan_repo_path=str(args.wan_repo_path),
            t5_max_length=int(args.t5_max_length),
            device=f"cuda:{device_id}",
        )
        processors_and_chunks.append((processor, chunk))

    all_results: list[tuple[str, bool]] = []
    with ProcessPoolExecutor(max_workers=len(processors_and_chunks)) as executor:
        futures = [executor.submit(process_t5_batch, item) for item in processors_and_chunks]
        for future in tqdm(futures, desc="Processing EgoVerse UMT5"):
            all_results.extend(future.result())

    failed = [path for path, ok in all_results if not ok]
    logger.info("EgoVerse UMT5 completed: success=%d failed=%d", len(all_results) - len(failed), len(failed))
    if failed:
        raise RuntimeError(f"UMT5 generation failed for {len(failed)} files, first failures: {failed[:5]}")


if __name__ == "__main__":
    main()
