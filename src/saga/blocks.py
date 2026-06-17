"""
saga.blocks
===========
Ready-made convolutional building blocks that use SAGA as their internal
activation function. These blocks can be used as drop-in replacements for
standard residual blocks in U-Net, ResNet, or EDSR style architectures.

Updated for v0.2.0: Safely unrolled to support dynamic tuple routing 
when interpretability gate extraction (return_gate=True) is activated globally.
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

    def forward(self, x: torch.Tensor):
        residual = self.shortcut(x)
        gates = []

        # --- First Convolution & Activation ---
        out1 = self.act1(self.bn1(self.conv1(x)))
        if isinstance(out1, tuple):  # Intercept tuple if interpretability is ON
            out, g1 = out1
            gates.append(g1)
        else:
            out = out1

        # --- Second Convolution & Final Activation ---
        out2 = self.act2(self.bn2(self.conv2(out)) + residual)
        if isinstance(out2, tuple):
            final_out, g2 = out2
            gates.append(g2)
        else:
            final_out = out2

        # --- Dynamic Return ---
        if gates:
            return final_out, gates
        return final_out


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

        # Unrolled from nn.Sequential to allow dynamic gate interception
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = SAGA(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = SAGA(bottleneck_channels)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out_act = SAGA(out_channels)

    def forward(self, x: torch.Tensor):
        gates = []

        # --- Block 1 ---
        out1 = self.act1(self.bn1(self.conv1(x)))
        if isinstance(out1, tuple):
            out, g = out1
            gates.append(g)
        else:
            out = out1
            
        # --- Block 2 ---
        out2 = self.act2(self.bn2(self.conv2(out)))
        if isinstance(out2, tuple):
            out, g = out2
            gates.append(g)
        else:
            out = out2
            
        # --- Block 3 & Skip Connection ---
        out3 = self.bn3(self.conv3(out))
        out_final = self.out_act(out3 + self.skip(x))
        if isinstance(out_final, tuple):
            final_tensor, g = out_final
            gates.append(g)
        else:
            final_tensor = out_final
            
        # --- Dynamic Return ---
        if gates:
            return final_tensor, gates
        return final_tensor