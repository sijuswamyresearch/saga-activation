"""
saga.utils
==========
Lightweight helpers for parameter accounting, gate control, and interpretability toggling.
"""

from __future__ import annotations
import torch.nn as nn
from .activation import SAGA

__all__ = ["count_parameters", "freeze_gate", "unfreeze_gate", "set_return_gate"]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Returns total parameters found in the target network architecture."""
    return sum(p.numel() for p in model.parameters() if (not trainable_only) or p.requires_grad)


def _set_gate_grad(model: nn.Module, requires_grad: bool) -> None:
    """Recursively targets all SAGA components to control gradient routing."""
    for module in model.modules():
        if isinstance(module, SAGA):
            # Freeze the entire spatial and gating pathway
            for name in ("spatial_conv", "spatial_bn", "gate_generator"):
                sub_module = getattr(module, name, None)
                if sub_module is not None:
                    for p in sub_module.parameters():
                        p.requires_grad_(requires_grad)


def freeze_gate(model: nn.Module) -> None:
    """
    Freeze all SAGA gating parameters in a model.
    Enables curriculum training sequences by isolating structural backbone tuning.
    """
    _set_gate_grad(model, False)


def unfreeze_gate(model: nn.Module) -> None:
    """Unfreeze all SAGA gating parameters in the target model infrastructure."""
    _set_gate_grad(model, True)


def set_return_gate(model: nn.Module, state: bool) -> None:
    """
    Recursively updates the return signature formatting for SAGA instances.
    
    Parameters
    ----------
    model : nn.Module
        Complete target network container.
    state : bool
        If True, layers return a tuple containing (output, gate_map).
        If False, layers function as drop-in tensor-to-tensor operations.
    """
    for module in model.modules():
        if isinstance(module, SAGA):
            module.return_gate = state