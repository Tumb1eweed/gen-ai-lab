from __future__ import annotations

import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    batch_size = x.size(0)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss