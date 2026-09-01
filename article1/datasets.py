"""Only MNIST, Fashion-MNIST and CIFAR-10 with deterministic evaluation views."""
from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def datasets_for(name: str, root: Path):
    root = str(root)
    if name == "cifar":
        eval_t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.4914, .4822, .4465), (.2470, .2435, .2616))])
        train_t = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), eval_t])
        return datasets.CIFAR10(root, train=True, download=True, transform=train_t), datasets.CIFAR10(root, train=True, download=True, transform=eval_t), datasets.CIFAR10(root, train=False, download=True, transform=eval_t)
    if name in {"mnist", "fmnist"}:
        cls = datasets.MNIST if name == "mnist" else datasets.FashionMNIST
        t = transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((.1307,) * 3, (.3081,) * 3)])
        return cls(root, train=True, download=True, transform=t), cls(root, train=True, download=True, transform=t), cls(root, train=False, download=True, transform=t)
    raise ValueError(f"unsupported Article-1 dataset {name!r}")


def labels_of(dataset) -> list[int]:
    return [int(value) for value in dataset.targets]


def test_loader(dataset, batch_size: int = 256) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
