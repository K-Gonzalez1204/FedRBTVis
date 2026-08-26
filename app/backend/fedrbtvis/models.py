import torch
from torch import nn

from fedrbtvis.config import ModelName


class TinyCNN(nn.Module):
    """Two convolution blocks, adaptive pooling, and one classifier."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(inputs))
        return self.classifier(torch.flatten(features, 1))


def build_model(name: ModelName, num_classes: int, seed: int) -> nn.Module:
    """Create a deterministically initialized model without pretrained weights."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if name == "tiny-cnn":
            return TinyCNN(num_classes)
        if name == "cifar-resnet18":
            from torchvision.models import resnet18

            model = resnet18(weights=None, num_classes=num_classes)
            model.conv1 = nn.Conv2d(
                3,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            model.maxpool = nn.Identity()
            return model
    raise ValueError(f"unknown model: {name}")
