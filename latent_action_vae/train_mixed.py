from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from latent_action_vae.cache_flows import cache_path
from latent_action_vae.datasets import AlohaLabeledDataset, EgoVerseUnlabeledDataset, collate_frame_pairs
from latent_action_vae.dcae_wrapper import DEFAULT_DCAE_MODEL, load_dcae
from latent_action_vae.flow_backend import build_flow_backend
from latent_action_vae.io import flow_xy_to_rgb, normalize_flow_rgb
from latent_action_vae.model import LatentActionVAEConfig, PaperLatentActionVAE, latent_action_vae_loss


class FlowRgbOnDemandDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        *,
        cache_root: Path,
        flow_backend,
        max_flow: float,
    ) -> None:
        self.base = base
        self.cache_root = cache_root
        self.flow_backend = flow_backend
        self.max_flow = float(max_flow)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.base[idx]
        output = cache_path(self.cache_root, item.source, item.sample_id)
        if output.exists():
            data = torch.load(output, map_location="cpu")
            flow_rgb = data["flow_rgb"].float()
        else:
            flow = self.flow_backend.compute(item.frame0, item.frame1)
            flow_rgb = normalize_flow_rgb(flow_xy_to_rgb(flow.unsqueeze(0), self.max_flow))[0].cpu().float()
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "flow_rgb": flow_rgb,
                    "flow_xy": flow.cpu().float(),
                    "sample_id": item.sample_id,
                    "flow_backend": getattr(self.flow_backend, "name", "unknown"),
                    "max_flow": self.max_flow,
                },
                output,
            )
        result = {"flow_rgb": flow_rgb, "sample_id": item.sample_id, "source": item.source}
        if item.action is not None:
            result["action"] = item.action.float()
        return result


def collate_flow_rgb(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "flow_rgb": torch.stack([item["flow_rgb"] for item in batch], dim=0),
        "sample_id": [item["sample_id"] for item in batch],
        "source": [item["source"] for item in batch],
    }
    if "action" in batch[0]:
        out["action"] = torch.stack([item["action"] for item in batch], dim=0)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Motus-style latent action VAE with 90/10 mixed data.")
    parser.add_argument("--egoverse-manifest", type=Path, default=Path("dataset/human_data/manifests/egoverse_unlabeled.jsonl"))
    parser.add_argument("--aloha-manifest", type=Path, default=Path("dataset/human_data/manifests/aloha_labeled.jsonl"))
    parser.add_argument("--cache-root", type=Path, default=Path("latent_action_vae/cache/flow_rgb"))
    parser.add_argument("--run-dir", type=Path, default=Path("latent_action_vae/runs/mixed_smoke"))
    parser.add_argument("--dcae-model", default=DEFAULT_DCAE_MODEL)
    parser.add_argument("--freeze-dcae", action="store_true")
    parser.add_argument("--flow-backend", choices=["ptlflow"], default="ptlflow")
    parser.add_argument("--ptlflow-model", default="dpflow")
    parser.add_argument("--ptlflow-ckpt", default="things")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-flow", type=float, default=20.0)
    parser.add_argument("--samples-per-video", type=int, default=8)
    parser.add_argument("--samples-per-episode", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--labeled-every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--beta", type=float, default=1.0e-4)
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, default=str, indent=2, sort_keys=True)

    flow_backend = build_flow_backend(
        args.flow_backend,
        model_name=args.ptlflow_model,
        ckpt_path=args.ptlflow_ckpt,
        device=args.device,
    )
    unlabeled = FlowRgbOnDemandDataset(
        EgoVerseUnlabeledDataset(
            args.egoverse_manifest,
            image_size=args.image_size,
            samples_per_video=args.samples_per_video,
            seed=args.seed,
        ),
        cache_root=args.cache_root,
        flow_backend=flow_backend,
        max_flow=args.max_flow,
    )
    labeled = FlowRgbOnDemandDataset(
        AlohaLabeledDataset(
            args.aloha_manifest,
            image_size=args.image_size,
            samples_per_episode=args.samples_per_episode,
            seed=args.seed,
        ),
        cache_root=args.cache_root,
        flow_backend=flow_backend,
        max_flow=args.max_flow,
    )

    unlabeled_loader = DataLoader(unlabeled, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_flow_rgb)
    labeled_loader = DataLoader(labeled, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_flow_rgb)
    unlabeled_iter = cycle(unlabeled_loader)
    labeled_iter = cycle(labeled_loader)

    device = torch.device(args.device)
    dcae = load_dcae(args.dcae_model, device=device, trainable=not args.freeze_dcae)
    config = LatentActionVAEConfig(hidden_dim=args.hidden_dim, action_dim=14)
    model = PaperLatentActionVAE(dcae=dcae, config=config, freeze_dcae=args.freeze_dcae).to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1.0e-4)

    log_path = args.run_dir / "train_log.jsonl"
    model.train()
    with log_path.open("a", encoding="utf-8") as log_f:
        for step in range(1, args.steps + 1):
            is_labeled = step % args.labeled_every == 0
            batch = next(labeled_iter if is_labeled else unlabeled_iter)
            flow_rgb = batch["flow_rgb"].to(device)
            robot_action = batch.get("action")
            if robot_action is not None:
                robot_action = robot_action.to(device)
            outputs = model(flow_rgb)
            losses = latent_action_vae_loss(
                flow_rgb,
                outputs,
                robot_action=robot_action,
                beta=args.beta,
                action_weight=args.action_weight,
            )
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            row = {
                "step": step,
                "kind": "labeled" if is_labeled else "unlabeled",
                "loss": float(losses["loss"].detach().cpu()),
                "recon_loss": float(losses["recon_loss"].detach().cpu()),
                "kl_loss": float(losses["kl_loss"].detach().cpu()),
                "action_loss": float(losses["action_loss"].detach().cpu()),
            }
            log_f.write(json.dumps(row, sort_keys=True) + "\n")
            log_f.flush()
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(
                    f"step={step} kind={row['kind']} loss={row['loss']:.6f} "
                    f"recon={row['recon_loss']:.6f} kl={row['kl_loss']:.6f} action={row['action_loss']:.6f}"
                )

    ckpt_path = args.run_dir / "latent_action_vae.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "latent_dim": 14,
                "token_dim": 512,
                "num_tokens": 4,
                "hidden_dim": args.hidden_dim,
                "dcae_model": args.dcae_model,
                "flow_backend": args.flow_backend,
                "ptlflow_model": args.ptlflow_model,
                "max_flow": args.max_flow,
                "camera_key": "observations/images/cam_high",
                "action_key": "action",
            },
        },
        ckpt_path,
    )
    print(f"saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
