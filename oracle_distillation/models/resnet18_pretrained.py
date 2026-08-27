"""ImageNet-pretrained ResNet-18 adapted to 32x32 CIFAR via *stem surgery*."""
from __future__ import annotations

import torch.nn as nn


def build_pretrained_resnet18(num_classes: int = 100) -> nn.Module:
    """Stock torchvision ResNet-18 (IMAGENET1K_V1) with a CIFAR stem + new head.

    Body weights (bn1, layer1..4) keep their pretrained values; conv1, maxpool and
    fc are replaced/reinitialised for 32x32 inputs and ``num_classes`` outputs.
    """
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # CIFAR stem surgery: 3x3 stride-1 conv, no max-pool (preserves 32x32 spatial res).
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    # Fresh classifier head.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
