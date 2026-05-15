from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_pool: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LaneCNN(nn.Module):
    """Small CNN for lane_out/lane_in ROI classification."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, use_pool=True),
            ConvBlock(32, 64, use_pool=True),
            ConvBlock(64, 128, use_pool=True),
            ConvBlock(128, 256, use_pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def create_model(name: str = "lane_cnn", num_classes: int = 2, dropout: float = 0.2) -> nn.Module:
    normalized_name = name.lower()
    if normalized_name in {"lane_cnn", "lanecnn"}:
        return LaneCNN(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unsupported model: {name}")

