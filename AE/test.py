from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dataset import build_dataloaders
from loss import reconstruction_loss
from model import ConvAutoencoder
from utils import configure_runtime, describe_device, ensure_dirs, resolve_device, save_image_grid


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained convolutional autoencoder on CIFAR-10 test data.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/ae_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/test"))
    parser.add_argument("--num-noise-samples", type=int, default=64)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    latent_dim = checkpoint["config"]["latent_dim"]

    device = resolve_device(args.device)
    configure_runtime(device)
    print(f"using device={describe_device(device)}")

    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, _, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_split=args.validation_split,
        seed=args.seed,
    )

    ensure_dirs(args.output_dir)
    total_test_loss = 0.0
    preview_batch = None

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            reconstructions, _ = model(images)
            loss = reconstruction_loss(reconstructions, images)
            total_test_loss += loss.item()
            if preview_batch is None:
                preview_batch = (images, reconstructions)

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"test_loss={avg_test_loss:.4f}")

    if preview_batch is not None:
        images, reconstructions = preview_batch
        comparison = torch.cat([images[:8], reconstructions[:8]], dim=0)
        save_image_grid(comparison.cpu(), args.output_dir / "test_reconstructions.png", nrow=8)
        print(f"saved test reconstructions to {args.output_dir / 'test_reconstructions.png'}")

    with torch.no_grad():
        noise_latents = torch.randn(args.num_noise_samples, latent_dim, device=device)
        noise_samples = model.decode(noise_latents)
        save_image_grid(noise_samples.cpu(), args.output_dir / "normal_noise_decode.png", nrow=8)
    print(f"saved normal-noise decoder samples to {args.output_dir / 'normal_noise_decode.png'}")


if __name__ == "__main__":
    main()