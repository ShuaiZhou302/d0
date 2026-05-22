from __future__ import annotations

import argparse
from pathlib import Path

import torch

from latent_action_vae.dcae_wrapper import DEFAULT_DCAE_MODEL, load_dcae
from latent_action_vae.io import flow_xy_to_rgb, load_flow_tensor, normalize_flow_rgb, resize_flow, save_latent_actions
from latent_action_vae.model import LatentActionVAEConfig, PaperLatentActionVAE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Motus-compatible 14D latent actions with DC-AE-token VAE.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help=".pt DPFlow tensor, [N,2,H,W] or [N,H,W,2]")
    parser.add_argument("--output", required=True, help="Output .pt latent action tensor, [N,14]")
    parser.add_argument("--dcae-model", default=None, help=f"Override DC-AE model, default from checkpoint or {DEFAULT_DCAE_MODEL}")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    device = torch.device(args.device)

    dcae_model = args.dcae_model or cfg.get("dcae_model", DEFAULT_DCAE_MODEL)
    dcae = load_dcae(dcae_model, device=device, trainable=False)
    model_cfg = LatentActionVAEConfig(
        latent_dim=int(cfg["latent_dim"]),
        token_dim=int(cfg["token_dim"]),
        num_tokens=int(cfg["num_tokens"]),
        hidden_dim=int(cfg["hidden_dim"]),
        action_dim=int(cfg["latent_dim"]),
    )
    model = PaperLatentActionVAE(dcae=dcae, config=model_cfg, freeze_dcae=True).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    flow = load_flow_tensor(args.input)
    flow = resize_flow(flow, int(cfg["input_size"]))
    flow_rgb = normalize_flow_rgb(flow_xy_to_rgb(flow, float(cfg["max_flow"]))).to(device)

    latents = []
    with torch.no_grad():
        for chunk in flow_rgb.split(64, dim=0):
            mu, _, _ = model.encode(chunk)
            latents.append(mu.cpu())
    latent_actions = torch.cat(latents, dim=0)
    save_latent_actions(latent_actions, args.output)
    print(f"saved latent actions: {Path(args.output)} shape={tuple(latent_actions.shape)}")


if __name__ == "__main__":
    main()
