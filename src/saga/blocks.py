"""
saga.blocks
===========
Ready-made convolutional building blocks that use SAGA as their internal
activation function. These blocks can be used as drop-in replacements for
standard residual blocks in U-Net, ResNet, or EDSR style architectures.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .activation import SAGA

__all__ = ["SAGAResBlock", "SAGABottleneck"]


class SAGAResBlock(nn.Module):
    """Residual block with SAGA activations."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = SAGA(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = SAGA(out_channels)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + residual)


class SAGABottleneck(nn.Module):
    """Bottleneck block (1x1 -> 3x3 -> 1x1) with SAGA activations."""

    def __init__(
        self,
        in_channels: int,
        bottleneck_channels: int | None = None,
        out_channels: int | None = None,
    ) -> None:
        super().__init__()
        bottleneck_channels = bottleneck_channels or max(in_channels // 4, 1)
        out_channels = out_channels or in_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            SAGA(bottleneck_channels),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            SAGA(bottleneck_channels),
            nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out_act = SAGA(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_act(self.net(x) + self.skip(x))