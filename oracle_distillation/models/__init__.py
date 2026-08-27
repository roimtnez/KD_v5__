from oracle_distillation.models.resnet9 import ResNet9
from oracle_distillation.models.resnet18 import ResNet18
from oracle_distillation.models.resnet18_pretrained import build_pretrained_resnet18
from oracle_distillation.models.mnist_net import MnistNet

__all__ = [
    "ResNet9", "ResNet18", "build_pretrained_resnet18", "MnistNet",
    "build_model", "dataset_default_arch",
]


def build_model(arch: str, num_classes: int = 10):
    arch = arch.lower()
    if arch == "resnet9":
        return ResNet9(num_classes=num_classes)
    if arch == "resnet18":
        return ResNet18(num_classes=num_classes)
    if arch == "resnet18_pretrained":
        return build_pretrained_resnet18(num_classes=num_classes)
    if arch == "mnist_net":
        return MnistNet(num_classes=num_classes)
    raise ValueError(f"Unknown architecture: {arch!r}")


def dataset_default_arch(dataset: str) -> str:
    from data.dataset_config import get_dataset_config
    return get_dataset_config(dataset).arch
