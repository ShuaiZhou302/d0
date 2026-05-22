from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from latent_action_vae.datasets import AlohaLabeledDataset, EgoVerseUnlabeledDataset, collate_frame_pairs
from latent_action_vae.flow_backend import build_flow_backend
from latent_action_vae.io import flow_xy_to_rgb, normalize_flow_rgb


def cache_path(cache_root: Path, source: str, sample_id: str) -> Path:
    safe = sample_id.replace("/", "__")
    return cache_root / source / f"{safe}.pt"


def save_flow_rgb(path: Path, flow: torch.Tensor, max_flow: float, backend_name: str, sample_id: str) -> None:
    flow_rgb = normalize_flow_rgb(flow_xy_to_rgb(flow.unsqueeze(0), max_flow=max_flow))[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "flow_rgb": flow_rgb.cpu().float(),
            "flow_xy": flow.cpu().float(),
            "sample_id": sample_id,
            "flow_backend": backend_name,
            "max_flow": float(max_flow),
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache flow-RGB tensors for latent action VAE training.")
    parser.add_argument("--egoverse-manifest", type=Path, default=Path("dataset/human_data/manifests/egoverse_unlabeled.jsonl"))
    parser.add_argument("--aloha-manifest", type=Path, default=Path("dataset/human_data/manifests/aloha_labeled.jsonl"))
    parser.add_argument("--cache-root", type=Path, default=Path("latent_action_vae/cache/flow_rgb"))
    parser.add_argument("--source", choices=["egoverse", "aloha", "both"], default="both")
    parser.add_argument("--flow-backend", choices=["ptlflow", "opencv"], default="ptlflow")
    parser.add_argument("--ptlflow-model", default="dpflow")
    parser.add_argument("--ptlflow-ckpt", default="things")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-flow", type=float, default=20.0)
    parser.add_argument("--samples-per-video", type=int, default=8)
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=0, help="0 means no explicit limit")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def iter_datasets(args: argparse.Namespace):
    if args.source in {"egoverse", "both"}:
        yield EgoVerseUnlabeledDataset(
            args.egoverse_manifest,
            image_size=args.image_size,
            samples_per_video=args.samples_per_video,
            seed=args.seed,
        )
    if args.source in {"aloha", "both"}:
        yield AlohaLabeledDataset(
            args.aloha_manifest,
            image_size=args.image_size,
            samples_per_episode=args.samples_per_episode,
            seed=args.seed,
        )


def main() -> None:
    args = parse_args()
    backend = build_flow_backend(
        args.flow_backend,
        model_name=args.ptlflow_model,
        ckpt_path=args.ptlflow_ckpt,
        device=args.device,
    )
    done = 0
    skipped = 0
    for dataset in iter_datasets(args):
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_frame_pairs)
        for batch in loader:
            sample_id = batch["sample_id"][0]
            source = batch["source"][0]
            output = cache_path(args.cache_root, source, sample_id)
            if output.exists():
                skipped += 1
                continue
            flow = backend.compute(batch["frame0"][0], batch["frame1"][0])
            save_flow_rgb(output, flow, args.max_flow, getattr(backend, "name", args.flow_backend), sample_id)
            done += 1
            if done == 1 or done % 100 == 0:
                print(f"cached={done} skipped={skipped} last={output}")
            if args.max_items and done >= args.max_items:
                print(f"finished cached={done} skipped={skipped}")
                return
    print(f"finished cached={done} skipped={skipped}")


if __name__ == "__main__":
    main()
