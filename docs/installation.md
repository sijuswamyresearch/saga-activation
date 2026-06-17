# Installation Guide

SAGA is designed to be lightweight, highly optimized, and easy to integrate into any existing PyTorch-based deep vision pipeline.

## Prerequisites

- **Python:** 3.10 or newer
- **PyTorch:** 2.0.0 or newer (CUDA support highly recommended)
- **Triton:** 2.1.0 or newer *(Optional, Linux/NVIDIA only. Required for fused kernel acceleration)*

## Installing via PyPI

The easiest way to install SAGA is directly from the Python Package Index. 

For standard CPU/GPU usage (Eager Mode):

```bash
pip install saga-activation
```
### For High-Speed GPU Execution (Linux & NVIDIA GPUs Only):

To unlock the fused memory-bandwidth optimizations, install SAGA with the Triton extension:

```bash
pip install "saga-activation[triton]"
```

## Installing from Source

If you are working in a standard local environment, clone the repository and install it in editable mode:

```bash
git clone [https://github.com/sijuswamyresearch/saga-activation.git](https://github.com/sijuswamyresearch/saga-activation.git)
cd saga-activation

# Standard installation
pip install -e .

# Installation with Triton acceleration and development tools
pip install -e ".[dev,triton]"
```
If you are testing SAGA in a notebook environment, you must use the shell prefix (!) and directory magic (%) to install the package directly within a cell:

```bash
!git clone [https://github.com/sijuswamyresearch/saga-activation.git](https://github.com/sijuswamyresearch/saga-activation.git)
%cd saga-activation
!pip install -e ".[triton]"
```

## Verifying the Installation

To verify that SAGA is installed and your GPU is picking it up correctly, run the following diagnostic script. This will also confirm if the new `v0.2.0` interpretability mode and Triton fallback are working.


```Python
import torch
from saga import SAGA

# Device-agnostic setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on: {device}")

# Initialize SAGA with interpretability (return_gate=True)
act = SAGA(in_channels=64, return_gate=True).to(device)
x = torch.randn(1, 64, 128, 128).to(device)

# Pass the tensor through the activation
out, gate = act(x)

print(f"Output shape: {out.shape}")      # Should be: torch.Size([1, 64, 128, 128])
print(f"Gate map shape: {gate.shape}")   # Should be: torch.Size([1, 64, 128, 128])
```
