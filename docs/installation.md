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

If you are working in a standard local environment, clone the repository and install it in editable mode:

```bash
git clone [https://github.com/sijuswamyresearch/saga-activation.git](https://github.com/sijuswamyresearch/saga-activation.git)
cd saga-activation
pip install -e .
```
If you are testing SAGA in a notebook environment, you must use the shell prefix (!) and directory magic (%) to install the package directly within a cell:

```bash
!git clone [https://github.com/sijuswamyresearch/saga-activation.git](https://github.com/sijuswamyresearch/saga-activation.git)
%cd saga-activation
!pip install -e .
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