"""
saga.utils
==========
Lightweight helpers for parameter accounting and gate control.
"""

from __future__ import annotations

import torch.nn as nn

from .activation import SAGA

__all__ = ["count_parameters", "freeze_gate", "unfreeze_gate"]


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the total number of (trainable) parameters in *model*.

    Parameters
    ----------
    model : nn.Module
    trainable_only : bool, optional
        When *True* (default) only count parameters with
        ``requires_grad == True``.

    Returns
    -------
    int
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if (not trainable_only) or p.requires_grad
    )


def _set_gate_grad(model: nn.Module, requires_grad: bool) -> None:
    """Recursively set requires_grad for all SAGA components."""
    for module in model.modules():
        if isinstance(module, SAGA):
            for name in ("spatial_conv", "spatial_bn", "gate_generator"):
                sub = getattr(module, name, None)
                if sub is not None:
                    for p in sub.parameters():
                        p.requires_grad_(requires_grad)


def freeze_gate(model: nn.Module) -> None:
    """Freeze all SAGA gating parameters in *model*.

    Useful for curriculum training: first train the backbone, then unfreeze
    the gates for fine-tuning.
    """
    _set_gate_grad(model, False)


def unfreeze_gate(model: nn.Module) -> None:
    """Unfreeze all SAGA gating parameters in *model*."""
    _set_gate_grad(model, True)