# SAGA — Spatially-Adaptive Gated Activation

[![CI](https://github.com/sijuswamyresearch/saga-activation/actions/workflows/ci.yml/badge.svg)](https://github.com/sijuswamyresearch/saga-activation/actions)
[![PyPI version](https://badge.fury.io/py/saga-activation.svg)](https://pypi.org/project/saga-activation/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/1258149652.svg)](https://doi.org/10.5281/zenodo.20582649)
[![Paper](https://img.shields.io/badge/Paper-Healthcare%20Analytics-blue)](https://doi.org/10.1016/j.health.2026.100468)

> **An Interpretable Deep Learning Method for Medical Image Deblurring and Restoration**
> Siju K.S., Vipin Venugopal, Mithun Kumar Kar, Jayakrishnan Anandakrishnan  
> *Healthcare Analytics* 9 (2026) 100468 · [doi:10.1016/j.health.2026.100468](https://doi.org/10.1016/j.health.2026.100468) 

---

## Overview

Standard activation functions (ReLU, SiLU, GELU) treat every spatial location in a feature map identically. However, in complex visual tasks and medical imaging, information content is *not* spatially uniform: structural boundaries carry high-frequency critical detail, while homogeneous regions require smooth suppression.

**SAGA (Version 0.2.0)** introduces an *adaptive residual activation block* that modulates the activation response position-by-position. It features **fused Triton kernels** for maximum GPU memory-bandwidth efficiency and built-in **dynamic gate extraction** for post-hoc interpretability.

1. **Context Extraction:** `T(X) = BN(W_s *3 X)` *(Depthwise 3x3 Convolution)*
2. **Residual Boost:** `B(X) = max(0, T(X) - X)`
3. **Gate Generation:** `G(X) = σ((W_g *1 T(X) + b_init) / τ)` *(Pointwise 1x1 Convolution with Temperature τ)*
4. **Output:** `SAGA(X) = X + (G(X) ⊙ B(X))`

This multi-path design lets the network selectively amplify high-frequency signals while smoothly gating uniform areas, acting as a high-speed, drop-in structural upgrade.

---


## Installation

Requirements: `Python ≥ 3.10`, `PyTorch ≥ 2.0`

```bash
pip install saga-activation
```

### High-Speed Fused GPU Execution (Linux & NVIDIA GPUs Only):

To unlock the fused memory-bandwidth optimizations via OpenAI Triton:

```python
pip install "saga-activation[triton]"
```

If you are working in a standard local environment, clone the repository and install it in editable mode:

```bash
git clone [https://github.com/sijuswamyresearch/saga-activation.git](https://github.com/sijuswamyresearch/saga-activation.git)
cd SAGA
pip install -e ".[dev,triton]"
```
>**Note:** If you are testing SAGA in a notebook environment, you must use the shell prefix (!) and directory magic (%) to install the package directly within a cell:

```bash
!git clone [https://github.com/sijuswamyresearch/saga-activation.git](https://github.com/sijuswamyresearch/saga-activation.git)
%cd saga-activation
!pip install -e .
```


## Quick Start
### Drop-in activation replacement

Because SAGA extracts spatial features, it requires the channel dimension of the incoming tensor upon initialization.

```python
import torch
from saga import SAGA

# Device-agnostic setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize SAGA with the number of incoming channels
act = SAGA(in_channels=64).to(device)          
x   = torch.randn(2, 64, 256, 256).to(device)  # (Batch, Channels, Height, Width)

# Forward pass executes via Triton (if available) and preserves tensor shape
y   = act(x)                                
print(y.shape) # Output shape: torch.Size([2, 64, 256, 256])
```

>**Note:** To ensure maximum compatibility across different environments (from CPU-only laptops to CUDA-enabled servers), we recommend using PyTorch's device-agnostic setup when initializing SAGA:

```bash
import torch
from saga import SAGA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
act = SAGA(in_channels=64).to(device)
x = torch.randn(1, 64, 128, 128).to(device)
print(f"Running on: {device}")
print(act(x).shape) # Should output: torch.Size([1, 64, 128, 128])
```
### Interpretability Mode (Extracting Gates)

SAGA allows you to extract the internal spatial gate maps for heatmap visualization or Gate Alignment Loss (GAL) training.

```python
# Enable return_gate=True and optionally adjust temperature/bias
act = SAGA(in_channels=64, return_gate=True, temperature=1.0).to(device)

out, gate_map = act(x)
print(out.shape)      # The activated tensor
print(gate_map.shape) # The spatial gating probabilities [0, 1]
```
### Global Interpretability Toggles & Pre-Built Blocks

SAGA includes unrolled, tuple-safe residual blocks and a global utility to turn interpretability on or off across your entire architecture with one line of code.

### Inside a U-Net or ResNet block

```python
from saga import SAGAResBlock, SAGABottleneck
from saga.utils import set_return_gate

# Build a network using SAGA blocks
model = torch.nn.Sequential(
    SAGAResBlock(64),
    SAGABottleneck(64, out_channels=128)
)

# Globally switch the entire model to return (output, gates) tuples!
set_return_gate(model, state=True)
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
