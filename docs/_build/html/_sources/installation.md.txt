# Installation Guide

SAGA is designed to be lightweight and easy to integrate into any existing PyTorch-based medical imaging pipeline.

## Prerequisites

- **Python:** 3.8 or newer
- **PyTorch:** 2.0.0 or newer (CUDA support highly recommended for large 3D medical volumes)

## Installing via PyPI

The easiest way to install SAGA is directly from the Python Package Index:

```bash
pip install saga-activation
```
## Installing from Source

If you want to modify the gating mechanism or contribute to the repository, you can install it from source in editable mode:

```bash
git clone [https://github.com/sijuswamyresearch/saga-activation.git](https://github.com/sijuswamyresearch/saga-activation.git)
cd saga-activation
pip install -e .
```

## Verifying the Installation
To verify that SAGA is installed and your GPU is picking it up correctly, run:

```bash
import torch
from saga import SAGA

act = SAGA(in_channels=64).cuda()
x = torch.randn(1, 64, 128, 128).cuda()
print(act(x).shape) # Should output: torch.Size([1, 64, 128, 128])
```
