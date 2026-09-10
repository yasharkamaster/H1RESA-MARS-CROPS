"""
Convolutional autoencoder for 227×227 HiRISE crops.

Maps each image to a fixed-length latent vector for Isolation Forest,
then reconstructs for error heatmaps. No pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Nearest upsample + two convs (avoids transpose checkerboard)."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    """
    1×227×227 → latent_dim → 1×227×227.

    Keeps a 14×14 feature map before the FC bottleneck so the decoder
    still has spatial layout (earlier global pools made reconstructions mushy).
    """

    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.feat_ch = 128
        self.feat_hw = 14

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d((0, 1, 0, 1)),  # 227 → 228
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, self.feat_ch, 1),
            nn.BatchNorm2d(self.feat_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

        flat = self.feat_ch * self.feat_hw * self.feat_hw
        self.enc_fc = nn.Linear(flat, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, flat)

        self.dec_body = nn.Sequential(
            _up_block(self.feat_ch, 128),
            _up_block(128, 64),
            _up_block(64, 32),
            _up_block(32, 16),
        )
        self.dec_head = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        nn.init.zeros_(self.dec_head[-2].weight)
        nn.init.constant_(self.dec_head[-2].bias, 0.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.enc_fc(h.flatten(1))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(z.size(0), self.feat_ch, self.feat_hw, self.feat_hw)
        h = self.dec_body(h)
        h = F.interpolate(h, size=(227, 227), mode="bilinear", align_corners=False)
        return self.dec_head(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


def ssim_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> torch.Tensor:
    """1 − SSIM with a Gaussian window (no pretrained nets)."""
    pad = window_size // 2
    channel = x.size(1)
    coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - pad
    g = torch.exp(-(coords**2) / (2 * 1.5**2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window = (window_1d @ window_1d.t()).expand(channel, 1, window_size, window_size).contiguous()

    mu_x = F.conv2d(x, window, padding=pad, groups=channel)
    mu_y = F.conv2d(y, window, padding=pad, groups=channel)
    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=pad, groups=channel) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=pad, groups=channel) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=pad, groups=channel) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return 1.0 - ssim_map.mean()


def grad_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L1 on finite-difference image gradients (edge term)."""
    dx_x = x[:, :, :, 1:] - x[:, :, :, :-1]
    dx_y = y[:, :, :, 1:] - y[:, :, :, :-1]
    dy_x = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy_y = y[:, :, 1:, :] - y[:, :, :-1, :]
    return F.l1_loss(dx_x, dx_y) + F.l1_loss(dy_x, dy_y)


def combined_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.55,
    beta: float = 0.30,
    gamma: float = 0.15,
) -> torch.Tensor:
    """alpha·MSE + beta·(1−SSIM) + gamma·∇L1."""
    mse = F.mse_loss(recon, target)
    if beta <= 0 and gamma <= 0:
        return mse
    loss = alpha * mse
    if beta > 0:
        loss = loss + beta * ssim_loss(recon, target)
    if gamma > 0:
        loss = loss + gamma * grad_loss(recon, target)
    return loss
