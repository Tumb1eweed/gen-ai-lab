from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid, save_image

from dataset import denormalize


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(device_index)
        return f"cuda:{device_index} ({name})"
    return "cpu"


def configure_runtime(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def save_checkpoint(path: Path, model, optimizer, epoch: int, config, history: dict[str, list[float]] | None = None, best_val_loss: float | None = None) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
            "history": history,
            "best_val_loss": best_val_loss,
        },
        path,
    )


def save_image_grid(images: torch.Tensor, path: Path, nrow: int = 8) -> None:
    grid = make_grid(denormalize(images), nrow=nrow)
    save_image(grid, path)


def save_loss_history(history: dict[str, list[float]], plot_dir: Path) -> None:
    history_path = plot_dir / "loss_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], label="train loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="validation loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("AE Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_curve.png")
    plt.close()


def load_loss_history(plot_dir: Path) -> dict[str, list[float]]:
    history_path = plot_dir / "loss_history.json"
    if not history_path.exists():
        return {"train_loss": [], "val_loss": []}

    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)
    return {
        "train_loss": list(history.get("train_loss", [])),
        "val_loss": list(history.get("val_loss", [])),
    }