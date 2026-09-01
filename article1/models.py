"""The three fixed Article-1 teacher/student architectures."""
from __future__ import annotations

import torch.nn as nn


class MnistNet(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, classes))

    def forward(self, x): return self.classifier(self.features(x))


class _Residual(nn.Module):
    def __init__(self, block): super().__init__(); self.block = block
    def forward(self, x): return x + self.block(x)


def _conv(a, b, pool=0):
    layers = [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(pool))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            _conv(3, 64), _conv(64, 128, 2), _Residual(nn.Sequential(_conv(128, 128), _conv(128, 128))),
            _conv(128, 256, 2), _conv(256, 512, 2), _Residual(nn.Sequential(_conv(512, 512), _conv(512, 512))),
            nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, classes),
        )
    def forward(self, x): return self.net(x)


def build_model(dataset: str, classes: int = 10) -> nn.Module:
    if dataset in {"mnist", "fmnist"}: return MnistNet(classes)
    if dataset == "cifar": return ResNet9(classes)
    raise ValueError(f"unsupported Article-1 dataset {dataset!r}")
