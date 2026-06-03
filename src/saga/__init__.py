"""
SAGA — Spatially-Adaptive Gated Activation for Medical Image Restoration
=========================================================================

Quick start
-----------
>>> import torch
>>> from saga import SAGA
>>> act = SAGA(in_channels=64)
>>> x   = torch.randn(2, 64, 256, 256)
>>> y   = act(x)          # same shape, spatially gated

Reference
---------
Siju K.S. et al. "An interpretable deep learning method for medical image
deblurring and restoration."  Healthcare Analytics 9 (2026) 100468.
https://doi.org/10.1016/j.health.2026.100468
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__: str = version("saga-activation")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .activation import SAGA, SAGALayer
from .blocks import SAGAResBlock, SAGABottleneck
from .utils import count_parameters, freeze_gate, unfreeze_gate

__all__ = [
    "SAGA",
    "SAGALayer",
    "SAGAResBlock",
    "SAGABottleneck",
    "count_parameters",
    "freeze_gate",
    "unfreeze_gate",
    "__version__",
]