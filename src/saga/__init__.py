"""
SAGA — Spatially-Adaptive Gated Activation for Medical Image Restoration
=========================================================================
An optimized, plug-and-play activation function featuring fused Triton kernels 
and dynamic gate map extraction capabilities for deep vision models.

Quick start
-----------
>>> import torch
>>> from saga import SAGA
>>> act = SAGA(in_channels=64, return_gate=False, temperature=1.0)
>>> x   = torch.randn(2, 64, 256, 256)
>>> y   = act(x)          # same shape, high-speed Triton execution

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
    __version__ = "0.2.0"

from .activation import SAGA, SAGALayer
from .blocks import SAGAResBlock, SAGABottleneck
from .utils import count_parameters, freeze_gate, unfreeze_gate, set_return_gate

__all__ = [
    "SAGA",
    "SAGALayer",
    "SAGAResBlock",
    "SAGABottleneck",
    "count_parameters",
    "freeze_gate",
    "unfreeze_gate",
    "set_return_gate",
    "__version__",
]