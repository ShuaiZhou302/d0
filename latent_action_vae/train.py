from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from latent_action_vae.dcae_wrapper import DEFAULT_DCAE_MODEL, load_dcae
from latent_action_vae.io import iter_flow_files, stack_limited_flow_rgbs
from latent_action_vae.model import LatentActionVAEConfig, PaperLatentActionVAE, latent_action_vae_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the paper-aligned DC-AE-token latent action VAE.")
    parser.add_argument("--input-dir", required=True, help="Directory containing DPFlow XY .pt tensors")
    parser.add_argument("--output", default="latent_action_vae/checkpoints/paper_latent_action_vae.pt")
    parser.add_argument("--dcae-model", default=DEFAULT_DCAE_MODEL)
    parser.add_argument("--freeze-dcae", action="store_true", help="Ablation only. Default is to finetune DC-AE.")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=14)
    parser.add_argument("--token-dim", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--max-flow", type=float, default=20.0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--beta", type=float, default=1.0e-4)
    parser.add_argument("--token-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = iter_flow_files(args.input_dir)
    flow_rgbs = stack_limited_flow_rgbs(files, args.max_items, args.input_size, args.max_flow)
    dataset = TensorDataset(flow_rgbs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    device = torch.device(args.device)
    dcae = load_dcae(args.dcae_model, device=device, trainable=not args.freeze_dcae)
    config = LatentActionVAEConfig(
        latent_dim=args.latent_dim,
        token_dim=args.token_dim,
        num_tokens=args.num_tokens,
        hidden_dim=args.hidden_dim,
        action_dim=args.latent_dim,
    )
    model = PaperLatentActionVAE(dcae=dcae, config=config, freeze_dcae=args.freeze_dcae).to(device)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1.0e-4)

    model.train()
    step = 0
    while step < args.steps:
        for (flow_rgb,) in loader:
            flow_rgb = flow_rgb.to(device)
            out = model(flow_rgb)
            losses = latent_action_vae_loss(
                flow_rgb,
                out,
                beta=args.beta,
                token_weight=args.token_weight,
            )
            optim.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step == 1 or step % 20 == 0 or step >= args.steps:
                print(
                    f"step={step} loss={losses['loss'].item():.6f} "
                    f"recon={losses['recon_loss'].item():.6f} "
                    f"kl={losses['kl_loss'].item():.6f} token={losses['token_loss'].item():.6f}"
                )
            if step >= args.steps:
                break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "latent_dim": args.latent_dim,
                "token_dim": args.token_dim,
                "num_tokens": args.num_tokens,
                "hidden_dim": args.hidden_dim,
                "max_flow": args.max_flow,
                "input_size": args.input_size,
                "dcae_model": args.dcae_model,
                "freeze_dcae": args.freeze_dcae,
                "flow_representation": "hsv_rgb",
            },
        },
        output,
    )
    print(f"saved checkpoint: {output}")


if __name__ == "__main__":
    main()
