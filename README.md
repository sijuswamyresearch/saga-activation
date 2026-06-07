# SAGA — Spatially-Adaptive Gated Activation

[![CI](https://github.com/sijuswamyresearch/saga-activation/actions/workflows/ci.yml/badge.svg)](https://github.com/sijuswamyresearch/saga-activation/actions)
[![PyPI version](https://badge.fury.io/py/saga-activation.svg)](https://pypi.org/project/saga-activation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/1258149652.svg)](https://doi.org/10.5281/zenodo.20582649)
[![Paper](https://img.shields.io/badge/Paper-Healthcare%20Analytics-blue)](https://doi.org/10.1016/j.health.2026.100468)

> **An Interpretable Deep Learning Method for Medical Image Deblurring and Restoration** > Siju K.S., Vipin Venugopal, Mithun Kumar Kar, Jayakrishnan Anandakrishnan  
> *Healthcare Analytics* 9 (2026) 100468 · [doi:10.1016/j.health.2026.100468](https://doi.org/10.1016/j.health.2026.100468) 

---

## Overview

Standard activation functions (ReLU, SiLU, GELU) treat every spatial location in a feature map identically. In medical images — such as CT slices and DXA scans — the information content is *not* spatially uniform: anatomical boundaries carry high-frequency diagnostically-critical detail, while homogeneous regions (background, soft tissue) require smooth suppression.

**SAGA** introduces an *adaptive residual activation block* that modulates the activation response position-by-position using highly efficient depthwise and pointwise convolutions:

1. **Context Extraction:** `T(X) = BN(W_s *3 X)` *(Depthwise 3x3 Convolution)*
2. **Gate Generation:** `G(X) = σ(W_g *1 T(X))` *(Pointwise 1x1 Convolution)*
3. **Residual Boost:** `B(X) = max(0, T(X) - X)`
4. **Output:** `SAGA(X) = X + (G(X) ⊙ B(X))`

This multi-path design lets the network selectively amplify high-frequency boundary signals while smoothly gating uniform background areas. It acts as a lightweight, drop-in structural upgrade that preserves spatial tensor dimensions without increasing the overall depth of the network.

---

## Installation

```bash
pip install saga-activation
```
Or install from source:
```bash
git clone [https://github.com/sijuswamyresearch/saga-activation.git](https://github.com/sijuswamyresearch/saga-activation.git)
cd SAGA
pip install -e ".[dev]"
```

Requirements: `Python ≥ 3.10`, `PyTorch ≥ 2.0`

## Quick Start
### Drop-in activation replacement

Because SAGA extracts spatial features, it requires the channel dimension of the incoming tensor upon initialization.

```python
import torch
from saga import SAGA

# Initialize SAGA with the number of incoming channels
act = SAGA(in_channels=64)          
x   = torch.randn(2, 64, 256, 256)  # (Batch, Channels, Height, Width)

# Forward pass preserves exact tensor shape
y   = act(x)                        # Output shape: (2, 64, 256, 256)
```

### Inside a U-Net or ResNet block

```python
import torch.nn as nn
from saga import SAGA

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SAGA(in_channels=out_ch),                              # ← swap in SAGA here
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SAGA(in_channels=out_ch),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.block(x))
```

### Pre-built residual blocks

```python
from saga import SAGAResBlock, SAGABottleneck

res    = SAGAResBlock(64)                         # standard residual block
bottle = SAGABottleneck(64, out_channels=128)     # bottleneck variant
```

### Gate curriculum training
For advanced optimization, you can freeze the spatial gates during the initial epochs to allow the main backbone weights to stabilize, then unfreeze them for fine-tuning.

```python
from saga.utils import freeze_gate, unfreeze_gate

# Phase 1 – train backbone only
freeze_gate(model)
train(model, epochs=10, lr=1e-3)

# Phase 2 – fine-tune gates
unfreeze_gate(model)
train(model, epochs=5, lr=1e-4)
```

## Repository Structure

```bash
SAGA/
├── src/
│   └── saga/                    # installable Python package
│       ├── __init__.py
│       ├── activation.py        # SAGA operator (core)
│       ├── blocks.py            # SAGAResBlock, SAGABottleneck
│       └── utils.py             # parameter counting, gate freeze helpers
│
├── tests/
│   ├── conftest.py
│   └── test_saga.py             # pytest suite (shapes, gradients, CPU/GPU)
├── docs/                        # Sphinx documentation source
├── pyproject.toml               # Build configuration
└── README.md
```

## Running the Tests

```python
pytest tests/ -v
```
## To run with coverage:

```python
pytest tests/ --cov=saga --cov-report=term-missing
```
## Citing
If SAGA is useful in your research, please cite:

```bash
@article{siju2026saga,
  title   = {An interpretable deep learning method for medical image deblurring and restoration},
  author  = {Siju K.S. and Vipin Venugopal and Mithun Kumar Kar and Jayakrishnan Anandakrishnan},
  journal = {Healthcare Analytics},
  volume  = {9},
  pages   = {100468},
  year    = {2026},
  doi     = {10.1016/j.health.2026.100468}
}
```
## License
MIT
---