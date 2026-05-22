from __future__ import annotations

from pathlib import Path

import torch

from latent_action_vae.io import flow_xy_to_rgb, make_synthetic_flows, normalize_flow_rgb, save_latent_actions
from latent_action_vae.model import FakeDCAE, LatentActionVAEConfig, PaperLatentActionVAE, latent_action_vae_loss


def main() -> None:
    out_dir = Path("latent_action_vae/smoke_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    flows = make_synthetic_flows(num_frames=16, size=256)
    torch.save({"flow": flows}, out_dir / "synthetic_flow.pt")
    flow_rgbs = normalize_flow_rgb(flow_xy_to_rgb(flows, max_flow=1.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcae = FakeDCAE().to(device)
    config = LatentActionVAEConfig(hidden_dim=256)
    model = PaperLatentActionVAE(dcae=dcae, config=config, freeze_dcae=False).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2.0e-3)
    flow_rgbs = flow_rgbs.to(device)

    model.train()
    for step in range(1, 11):
        idx = torch.randint(0, flow_rgbs.shape[0], (2,), device=device)
        batch = flow_rgbs[idx]
        out = model(batch)
        losses = latent_action_vae_loss(batch, out, beta=1.0e-4)
        optim.zero_grad(set_to_none=True)
        losses["loss"].backward()
        optim.step()
        if step in {1, 5, 10}:
            print(
                f"step={step} loss={losses['loss'].item():.6f} "
                f"recon={losses['recon_loss'].item():.6f} kl={losses['kl_loss'].item():.6f}"
            )

    model.eval()
    with torch.no_grad():
        mu, _, tokens = model.encode(flow_rgbs)
        out = model(flow_rgbs[:2])
    save_latent_actions(mu, out_dir / "latent_action_dim14" / "synthetic.pt")

    ckpt_path = out_dir / "paper_latent_action_vae.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "latent_dim": 14,
                "token_dim": 512,
                "num_tokens": 4,
                "hidden_dim": 256,
                "max_flow": 1.0,
                "input_size": 256,
                "dcae_model": "fake-dcae-smoke-test",
                "freeze_dcae": False,
                "flow_representation": "hsv_rgb",
            },
        },
        ckpt_path,
    )
    print(f"checkpoint={ckpt_path}")
    print(f"tokens_shape={tuple(tokens.shape)}")
    print(f"latent_shape={tuple(mu.shape)}")
    print(f"recon_shape={tuple(out['recon'].shape)}")
    assert tuple(tokens.shape) == (16, 4, 512)
    assert tuple(mu.shape) == (16, 14)
    assert tuple(out["recon"].shape) == (2, 3, 256, 256)
    assert torch.isfinite(mu).all()
    print("smoke test passed")


if __name__ == "__main__":
    main()
