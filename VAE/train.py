from __future__ import annotations

from pathlib import Path

import torch

from config import get_config
from dataset import build_dataloaders
from loss import vae_loss
from model import ConvVAE
from utils import (
    configure_runtime,
    describe_device,
    ensure_dirs,
    load_loss_history,
    resolve_device,
    save_checkpoint,
    save_image_grid,
    save_loss_history,
    set_seed,
)


def evaluate_and_save(model, batch, epoch: int, reconstruction_dir: Path, sample_dir: Path, device) -> None:
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        reconstructions, _, _ = model(batch)
        comparison = torch.cat([batch[:8], reconstructions[:8]], dim=0)
        save_image_grid(comparison.cpu(), reconstruction_dir / f"epoch_{epoch:03d}.png", nrow=8)

        samples = model.sample(num_samples=64, device=device)
        save_image_grid(samples.cpu(), sample_dir / f"epoch_{epoch:03d}.png", nrow=8)
    model.train()


def train() -> None:
    config = get_config()
    device = resolve_device(config.device)
    set_seed(config.seed)
    configure_runtime(device)
    ensure_dirs(
        config.output_dir,
        config.checkpoint_dir,
        config.plot_dir,
        config.reconstruction_dir,
        config.sample_dir,
    )
    print(f"using device={describe_device(device)}")

    train_loader, test_loader = build_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    model = ConvVAE(latent_dim=config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = load_loss_history(config.plot_dir)
    start_epoch = 1

    if config.resume_checkpoint is not None:
        checkpoint = torch.load(config.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        checkpoint_history = checkpoint.get("history")
        if checkpoint_history:
            history = {
                "total_loss": list(checkpoint_history.get("total_loss", [])),
                "recon_loss": list(checkpoint_history.get("recon_loss", [])),
                "kl_loss": list(checkpoint_history.get("kl_loss", [])),
            }
        print(f"resuming from checkpoint={config.resume_checkpoint} at epoch={start_epoch:03d}")

    preview_batch, _ = next(iter(test_loader))

    for epoch in range(start_epoch, config.epochs + 1):
        total_loss_sum = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)

            reconstructions, mu, logvar = model(images)
            total_loss, recon_loss, kl_loss = vae_loss(
                reconstructions,
                images,
                mu,
                logvar,
                beta=config.beta,
            )
            total_loss.backward()
            optimizer.step()

            total_loss_sum += total_loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()

        num_batches = len(train_loader)
        avg_total = total_loss_sum / num_batches
        avg_recon = recon_loss_sum / num_batches
        avg_kl = kl_loss_sum / num_batches
        print(
            f"epoch={epoch:03d} total_loss={avg_total:.4f} recon_loss={avg_recon:.4f} kl_loss={avg_kl:.4f}"
        )

        history["total_loss"].append(avg_total)
        history["recon_loss"].append(avg_recon)
        history["kl_loss"].append(avg_kl)

        save_checkpoint(config.checkpoint_dir / "vae_last.pt", model, optimizer, epoch, config, history=history)
        save_loss_history(history, config.plot_dir)
        if epoch % config.sample_interval == 0:
            evaluate_and_save(
                model=model,
                batch=preview_batch,
                epoch=epoch,
                reconstruction_dir=config.reconstruction_dir,
                sample_dir=config.sample_dir,
                device=device,
            )


if __name__ == "__main__":
    train()