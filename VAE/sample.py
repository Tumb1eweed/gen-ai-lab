from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import ConvVAE
from utils import configure_runtime, describe_device, ensure_dirs, resolve_device, save_image_grid


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample images from a trained VAE decoder.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", type=Path, default=Path("outputs/samples/manual_sample.png"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    latent_dim = checkpoint["config"]["latent_dim"]
    model = ConvVAE(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = resolve_device(args.device)
    configure_runtime(device)
    model = model.to(device)
    model.eval()
    print(f"using device={describe_device(device)}")

    ensure_dirs(args.output.parent)
    with torch.no_grad():
        samples = model.sample(num_samples=args.num_samples, device=device)
        save_image_grid(samples.cpu(), args.output, nrow=8)

    print(f"saved samples to {args.output}")


if __name__ == "__main__":
    main()