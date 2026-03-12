from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
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


def build_dataloaders(data_dir, batch_size: int, num_workers: int, validation_split: float, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    transform = build_transforms()
    pin_memory = torch.cuda.is_available()

    full_train_dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, transform=transform, download=True)

    val_size = int(len(full_train_dataset) * validation_split)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    return train_loader, val_loader, test_loader


def denormalize(images):
    return (images * 0.5 + 0.5).clamp(0.0, 1.0)