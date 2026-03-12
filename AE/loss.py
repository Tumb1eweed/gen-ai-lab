from __future__ import annotations

import torch.nn.functional as F


def reconstruction_loss(recon_x, x):
    batch_size = x.size(0)
    return F.mse_loss(recon_x, x, reduction="sum") / batch_size