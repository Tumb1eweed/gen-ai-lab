from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"


@dataclass
class TrainConfig:
    data_dir: Path = ROOT_DIR / "data"
    output_dir: Path = OUTPUT_DIR
    checkpoint_dir: Path = OUTPUT_DIR / "checkpoints"
    plot_dir: Path = OUTPUT_DIR / "plots"
    reconstruction_dir: Path = OUTPUT_DIR / "reconstructions"
    sample_dir: Path = OUTPUT_DIR / "samples"
    batch_size: int = 128
    epochs: int = 100
    learning_rate: float = 1e-3
    beta: float = 1.0
    latent_dim: int = 128
    num_workers: int = 2
    image_size: int = 32
    sample_interval: int = 1
    num_preview_images: int = 8
    device: str = "auto"
    seed: int = 42
    resume_checkpoint: Optional[Path] = None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a convolutional VAE on CIFAR-10.")
    parser.add_argument("--data-dir", type=Path, default=TrainConfig.data_dir)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--beta", type=float, default=TrainConfig.beta)
    parser.add_argument("--latent-dim", type=int, default=TrainConfig.latent_dim)
    parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    parser.add_argument("--sample-interval", type=int, default=TrainConfig.sample_interval)
    parser.add_argument("--num-preview-images", type=int, default=TrainConfig.num_preview_images)
    parser.add_argument("--device", type=str, default=TrainConfig.device, choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--resume-checkpoint", type=Path, default=TrainConfig.resume_checkpoint)
    return parser


def get_config() -> TrainConfig:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=output_dir,
        checkpoint_dir=output_dir / "checkpoints",
        plot_dir=output_dir / "plots",
        reconstruction_dir=output_dir / "reconstructions",
        sample_dir=output_dir / "samples",
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        latent_dim=args.latent_dim,
        num_workers=args.num_workers,
        sample_interval=args.sample_interval,
        num_preview_images=args.num_preview_images,
        device=args.device,
        seed=args.seed,
        resume_checkpoint=args.resume_checkpoint,
    )