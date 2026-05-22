from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LatentActionVAEConfig:
    latent_dim: int = 14
    token_dim: int = 512
    num_tokens: int = 4
    hidden_dim: int = 1024
    encoder_depth: int = 2
    decoder_depth: int = 2
    action_dim: int = 14


class PaperLatentActionVAE(nn.Module):
    """Motus-style latent action VAE over DC-AE flow-RGB tokens.

    Pipeline:
      flow RGB [B, 3, 256, 256]
        -> trainable DC-AE encoder
        -> tokens [B, 4, 512]
        -> lightweight encoder
        -> latent action z [B, 14]
        -> lightweight decoder
        -> reconstructed tokens [B, 4, 512]
        -> trainable DC-AE decoder
        -> reconstructed flow RGB [B, 3, 256, 256]

    The DC-AE module is intentionally trainable by default. If a caller wants to
    freeze it for an ablation, pass `freeze_dcae=True`.
    """

    def __init__(
        self,
        dcae: nn.Module,
        config: LatentActionVAEConfig | None = None,
        freeze_dcae: bool = False,
    ) -> None:
        super().__init__()
        self.dcae = dcae
        self.config = config or LatentActionVAEConfig()
        self.freeze_dcae = bool(freeze_dcae)

        if self.freeze_dcae:
            for param in self.dcae.parameters():
                param.requires_grad_(False)

        flat_dim = self.config.num_tokens * self.config.token_dim
        self.encoder = build_mlp(
            in_dim=flat_dim,
            hidden_dim=self.config.hidden_dim,
            out_dim=self.config.hidden_dim,
            depth=self.config.encoder_depth,
        )
        self.mu = nn.Linear(self.config.hidden_dim, self.config.latent_dim)
        self.logvar = nn.Linear(self.config.hidden_dim, self.config.latent_dim)

        self.decoder = build_mlp(
            in_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            out_dim=flat_dim,
            depth=self.config.decoder_depth,
        )
        self.action_head = nn.Linear(self.config.latent_dim, self.config.action_dim)

    def encode_tokens(self, flow_rgb: torch.Tensor) -> torch.Tensor:
        latent = self.dcae.encode(flow_rgb)
        return dcae_latent_to_tokens(latent, self.config.num_tokens, self.config.token_dim)

    def encode(self, flow_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.encode_tokens(flow_rgb)
        hidden = self.encoder(tokens.flatten(1))
        mu = self.mu(hidden)
        logvar = self.logvar(hidden).clamp(-12.0, 8.0)
        return mu, logvar, tokens

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_tokens(self, z: torch.Tensor) -> torch.Tensor:
        token_flat = self.decoder(z)
        return token_flat.view(z.shape[0], self.config.num_tokens, self.config.token_dim)

    def decode_flow_rgb(self, tokens: torch.Tensor) -> torch.Tensor:
        latent = tokens_to_dcae_latent(tokens)
        return self.dcae.decode(latent)

    def forward(self, flow_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar, tokens = self.encode(flow_rgb)
        z = self.reparameterize(mu, logvar)
        recon_tokens = self.decode_tokens(z)
        recon_flow_rgb = self.decode_flow_rgb(recon_tokens)
        pred_action = self.action_head(z)
        return {
            "recon": recon_flow_rgb,
            "recon_tokens": recon_tokens,
            "tokens": tokens,
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "pred_action": pred_action,
        }


def build_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int) -> nn.Sequential:
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    layers: list[nn.Module] = []
    cur_dim = in_dim
    for _ in range(depth - 1):
        layers.extend(
            [
                nn.Linear(cur_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ]
        )
        cur_dim = hidden_dim
    layers.append(nn.Linear(cur_dim, out_dim))
    return nn.Sequential(*layers)


def dcae_latent_to_tokens(latent: torch.Tensor, num_tokens: int = 4, token_dim: int = 512) -> torch.Tensor:
    """Convert DC-AE f128c512 latent [B,512,2,2] to [B,4,512]."""
    if latent.dim() != 4:
        raise ValueError(f"Expected DC-AE latent [B,C,H,W], got {tuple(latent.shape)}")
    bsz, channels, height, width = latent.shape
    if channels != token_dim:
        raise ValueError(f"Expected token_dim={token_dim}, got latent channels={channels}")
    if height * width != num_tokens:
        raise ValueError(f"Expected {num_tokens} spatial tokens, got H*W={height * width}")
    return latent.flatten(2).transpose(1, 2).contiguous()


def tokens_to_dcae_latent(tokens: torch.Tensor) -> torch.Tensor:
    """Convert [B,4,512] tokens back to [B,512,2,2]."""
    if tokens.dim() != 3:
        raise ValueError(f"Expected tokens [B,N,C], got {tuple(tokens.shape)}")
    bsz, num_tokens, token_dim = tokens.shape
    spatial = int(num_tokens**0.5)
    if spatial * spatial != num_tokens:
        raise ValueError(f"num_tokens must be a square for DC-AE spatial map, got {num_tokens}")
    return tokens.transpose(1, 2).contiguous().view(bsz, token_dim, spatial, spatial)


def latent_action_vae_loss(
    flow_rgb: torch.Tensor,
    outputs: dict[str, torch.Tensor],
    *,
    robot_action: Optional[torch.Tensor] = None,
    beta: float = 1.0e-4,
    action_weight: float = 1.0,
    token_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(outputs["recon"], flow_rgb)
    kl_loss = -0.5 * torch.mean(1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp())

    if token_weight > 0:
        token_loss = F.mse_loss(outputs["recon_tokens"], outputs["tokens"].detach())
    else:
        token_loss = flow_rgb.new_tensor(0.0)

    if robot_action is not None:
        action_loss = F.mse_loss(outputs["pred_action"], robot_action.to(outputs["pred_action"].dtype))
    else:
        action_loss = flow_rgb.new_tensor(0.0)

    total = recon_loss + float(beta) * kl_loss + float(token_weight) * token_loss
    if robot_action is not None:
        total = total + float(action_weight) * action_loss

    return {
        "loss": total,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "token_loss": token_loss,
        "action_loss": action_loss,
    }


class FakeDCAE(nn.Module):
    """Tiny DC-AE-shaped module for CPU smoke tests only."""

    def __init__(self, token_dim: int = 512) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 4),
            nn.SiLU(),
            nn.Conv2d(64, 256, 4, 4),
            nn.SiLU(),
            nn.Conv2d(256, token_dim, 8, 8),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(token_dim, 256, 8, 8),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 64, 4, 4),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 3, 4, 4),
            nn.Tanh(),
        )

    @property
    def spatial_compression_ratio(self) -> int:
        return 128

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)
