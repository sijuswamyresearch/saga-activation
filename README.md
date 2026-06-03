# SAGA — Spatially-Adaptive Gated Activation

[![CI](https://github.com/sijuswamyresearch/SAGA/actions/workflows/ci.yml/badge.svg)](https://github.com/sijuswamyresearch/SAGA/actions)
[![PyPI version](https://badge.fury.io/py/saga-activation.svg)](https://pypi.org/project/saga-activation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Paper](https://img.shields.io/badge/Paper-Healthcare%20Analytics-blue)](https://doi.org/10.1016/j.health.2026.100468)

> **An Interpretable Deep Learning Method for Medical Image Deblurring and Restoration**  
> Siju K.S., Vipin Venugopal, Mithun Kumar Kar, Jayakrishnan Anandakrishnan  
> *Healthcare Analytics* 9 (2026) 100468 · [doi:10.1016/j.health.2026.100468](https://doi.org/10.1016/j.health.2026.100468)

---

## Overview

Standard activation functions (ReLU, SiLU, GELU) treat every spatial location in a feature map identically.  In medical images — CT slices, DXA scans — the information content is *not* spatially uniform: anatomical boundaries carry high-frequency diagnostically-critical detail while homogeneous regions (background, soft tissue) require smooth suppression.

**SAGA** introduces a *learned spatial gating map* that modulates the activation response position-by-position:

```
G(X)    = σ(W_g * X)        # spatial gate (depthwise-separable conv)
SAGA(X) = G(X) ⊙ φ(X)      # φ = SiLU (default)
```

This two-path design lets the network selectively amplify high-frequency boundary signals while smoothly gating uniform background areas — without increasing the depth of the network.

---

## Installation

```bash
pip install saga-activation
```

Or install from source:

```bash
git clone https://github.com/sijuswamyresearch/SAGA.git
cd SAGA
pip install -e ".[dev]"
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0

---

## Quick Start

### Drop-in activation replacement

```python
import torch
from saga import SAGA

# Replace any activation layer with SAGA
act = SAGA(in_channels=64)          # matches the channel dim of your feature map
x   = torch.randn(2, 64, 256, 256)  # (B, C, H, W)
y   = act(x)                         # same shape: (2, 64, 256, 256)
```

### Inside a U-Net encoder block

```python
import torch.nn as nn
from saga import SAGA

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SAGA(out_ch),                          # ← swap in SAGA here
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            SAGA(out_ch),
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

### Base-activation variants

```python
from saga import SAGA

act_relu = SAGA(64, base_activation="relu")
act_gelu = SAGA(64, base_activation="gelu")
act_tanh = SAGA(64, base_activation="tanh")
```

### Gate curriculum training

```python
from saga.utils import freeze_gate, unfreeze_gate

# Phase 1 – train backbone only
freeze_gate(model)
train(model, epochs=10, lr=1e-3)

# Phase 2 – fine-tune gates
unfreeze_gate(model)
train(model, epochs=5, lr=1e-4)
```

---

## Repository Structure

```
SAGA/
├── saga/                        # installable Python package
│   ├── __init__.py
│   ├── activation.py            # SAGA operator (core)
│   ├── blocks.py                # SAGAResBlock, SAGABottleneck
│   └── utils.py                 # parameter counting, gate freeze helpers
│
├── tests/
│   ├── conftest.py
│   └── test_saga.py             # pytest suite (shapes, edge cases, GPU, gradients)
│
├── SAGA_Supplementary_Code/     # original experimental pipeline
│   ├── models/
│   │   ├── saga_layer.py        # raw research implementation
│   │   ├── unet.py
│   │   ├── resnet.py
│   │   ├── edsr.py
│   │   └── vggnet.py
│   ├── generate_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   ├── xai_analysis.py
│   └── clinical_validation.py
│
├── docs/                        # Sphinx documentation source
├── .github/workflows/ci.yml     # GitHub Actions CI
├── pyproject.toml
└── README.md
```

---

## Experimental Results (summary)

| Model         | Activation | CT PSNR (dB) | CT SSIM | DXA PSNR (dB) | DXA SSIM |
|---------------|-----------|:------------:|:-------:|:-------------:|:--------:|
| U-Net         | ReLU      | 32.14        | 0.891   | 30.87         | 0.873    |
| U-Net         | SiLU      | 33.01        | 0.902   | 31.54         | 0.881    |
| **U-Net**     | **SAGA**  | **34.67**    | **0.921** | **33.12**   | **0.903** |
| DeblurResNet  | ReLU      | 31.89        | 0.883   | 30.21         | 0.864    |
| **DeblurResNet** | **SAGA** | **34.11** | **0.916** | **32.78**   | **0.897** |

Full results and ablation studies are reported in the paper.

---

## Running the Tests

```bash
pytest tests/ -v
```

To run with coverage:

```bash
pytest tests/ --cov=saga --cov-report=term-missing
```

---

## Citing

If SAGA is useful in your research, please cite:

```bibtex
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

---

## License

[MIT](LICENSE)
