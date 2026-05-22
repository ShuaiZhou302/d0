from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BaselineConvFlowVAE(nn.Module):
    """Temporary smoke-test baseline.

    This is intentionally kept separate from the paper-aligned latent-action VAE
    in `model.py`. It should not be used as the official Motus latent-action
    generator.
    """

    def __init__(
        self,
        latent_dim: int = 14,
        input_channels: int = 3,
        base_channels: int = 32,
        input_size: int = 64,
    ) -> None:
        super().__init__()
        if input_size % 16 != 0:
            raise ValueError(f"input_size must be divisible by 16, got {input_size}")

        self.latent_dim = int(latent_dim)
        self.input_channels = int(input_channels)
        self.base_channels = int(base_channels)
        self.input_size = int(input_size)
        self.spatial = input_size // 16
        hidden_channels = base_channels * 8
        flat_dim = hidden_channels * self.spatial * self.spatial

        self.encoder = nn.Sequential(
            conv_block(input_channels, base_channels),
            conv_block(base_channels, base_channels * 2),
            conv_block(base_channels * 2, base_channels * 4),
            conv_block(base_channels * 4, hidden_channels),
        )
        self.mu = nn.Linear(flat_dim, latent_dim)
        self.logvar = nn.Linear(flat_dim, latent_dim)

        self.decoder_in = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            deconv_block(hidden_channels, base_channels * 4),
            deconv_block(base_channels * 4, base_channels * 2),
            deconv_block(base_channels * 2, base_channels),
            nn.ConvTranspose2d(base_channels, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode(self, flow_rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(flow_rgb)
        flat = features.flatten(1)
        return self.mu(flat), self.logvar(flat).clamp(-12.0, 8.0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_in(z)
        x = x.view(z.shape[0], self.base_channels * 8, self.spatial, self.spatial)
        return self.decoder(x)

    def forward(self, flow_rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(flow_rgb)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return {"recon": recon, "z": z, "mu": mu, "logvar": logvar}


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.SiLU(inplace=True),
    )


def deconv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
        nn.SiLU(inplace=True),
    )


def baseline_vae_loss(
    flow_rgb: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    recon_loss = F.mse_loss(recon, flow_rgb)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + float(beta) * kl
    return {"loss": total, "recon_loss": recon_loss, "kl_loss": kl}
