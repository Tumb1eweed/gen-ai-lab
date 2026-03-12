from __future__ import annotations

from pathlib import Path

import torch

from config import get_config
from dataset import build_dataloaders
from loss import reconstruction_loss
from model import ConvAutoencoder
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


def save_validation_reconstructions(model, batch, epoch: int, reconstruction_dir: Path, device) -> None:
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        reconstructions, _ = model(batch)
        comparison = torch.cat([batch[:8], reconstructions[:8]], dim=0)
        save_image_grid(comparison.cpu(), reconstruction_dir / f"epoch_{epoch:03d}.png", nrow=8)
    model.train()


def run_validation(model, val_loader, device) -> float:
    model.eval()
    loss_sum = 0.0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            reconstructions, _ = model(images)
            loss = reconstruction_loss(reconstructions, images)
            loss_sum += loss.item()
    model.train()
    return loss_sum / len(val_loader)


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
        config.test_dir,
    )
    print(f"using device={describe_device(device)}")

    train_loader, val_loader, _ = build_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        validation_split=config.validation_split,
        seed=config.seed,
    )

    model = ConvAutoencoder(latent_dim=config.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history = load_loss_history(config.plot_dir)
    start_epoch = 1
    best_val_loss = float("inf")

    if config.resume_checkpoint is not None:
        checkpoint = torch.load(config.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        checkpoint_history = checkpoint.get("history")
        if checkpoint_history:
            history = {
                "train_loss": list(checkpoint_history.get("train_loss", [])),
                "val_loss": list(checkpoint_history.get("val_loss", [])),
            }
        print(f"resuming from checkpoint={config.resume_checkpoint} at epoch={start_epoch:03d}")

    preview_batch, _ = next(iter(val_loader))

    for epoch in range(start_epoch, config.epochs + 1):
        train_loss_sum = 0.0

        for images, _ in train_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)

            reconstructions, _ = model(images)
            loss = reconstruction_loss(reconstructions, images)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_val_loss = run_validation(model, val_loader, device)
        print(f"epoch={epoch:03d} train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        save_checkpoint(
            config.checkpoint_dir / "ae_last.pt",
            model,
            optimizer,
            epoch,
            config,
            history=history,
            best_val_loss=best_val_loss,
        )
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                config.checkpoint_dir / "ae_best.pt",
                model,
                optimizer,
                epoch,
                config,
                history=history,
                best_val_loss=best_val_loss,
            )

        save_loss_history(history, config.plot_dir)
        if epoch % config.sample_interval == 0:
            save_validation_reconstructions(model, preview_batch, epoch, config.reconstruction_dir, device)


if __name__ == "__main__":
    train()