"""
saga.activation
===============
Spatially-Adaptive Gated Activation (SAGA) operator.

SAGA extracts spatial context via a depthwise convolution, calculates a 
residual boost, and dynamically gates this boost before adding it back to 
the original input. This enables the network to selectively route gradient 
flow through high-frequency anatomical boundary regions.

Reference
---------
Siju K.S., Venugopal V., Kar M.K., Anandakrishnan J.
"An interpretable deep learning method for medical image deblurring and
restoration." Healthcare Analytics 9 (2026) 100468.
https://doi.org/10.1016/j.health.2026.100468
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SAGA", "SAGALayer"]


class SAGA(nn.Module):
    """Spatially-Adaptive Gated Activation (SAGA).

    Parameters
    ----------
    in_channels : int
        Number of input (and output) channels.

    Examples
    --------
    >>> import torch
    >>> from saga import SAGA
    >>> act = SAGA(in_channels=64)
    >>> x = torch.randn(2, 64, 32, 32)
    >>> y = act(x)
    >>> y.shape
    torch.Size([2, 64, 32, 32])
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        
        # Spatial context extractor
        self.spatial_conv = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=3, 
            padding=1, 
            groups=in_channels, 
            bias=False
        )
        self.spatial_bn = nn.BatchNorm2d(in_channels)
        
        # Dynamic gate generator
        self.gate_generator = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=1, 
            padding=0, 
            bias=True
        )
        
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights to ensure stable early-stage training."""
        nn.init.kaiming_normal_(self.spatial_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.spatial_bn.weight, 1)
        nn.init.constant_(self.spatial_bn.bias, 0)
        
        # Gate generator initialization
        nn.init.constant_(self.gate_generator.weight, 0)
        # Starts gate near ~0.88 for stability during initial epochs
        nn.init.constant_(self.gate_generator.bias, 2.0) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, C, H, W)`` with ``C == in_channels``.

        Returns
        -------
        torch.Tensor
            Same shape as *x*.
        """
        # Extract spatial context
        T_x = self.spatial_bn(self.spatial_conv(x))
        
        # Calculate positive boost
        boost = F.relu(T_x - x)
        
        # Generate spatial gate
        gate = torch.sigmoid(self.gate_generator(T_x))
        
        # Add gated boost to identity
        return x + (gate * boost)

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}"


# Alias for drop-in use inside sequential blocks
SAGALayer = SAGA