from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ]
    )


def build_dataloaders(data_dir, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    transform = build_transforms()
    pin_memory = torch.cuda.is_available()
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def denormalize(images):
    return (images * 0.5 + 0.5).clamp(0.0, 1.0)