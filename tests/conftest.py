"""
tests/conftest.py
=================
Shared testing fixtures for the SAGA activation package.
"""

import pytest
import torch

@pytest.fixture(scope="session", autouse=True)
def set_deterministic_seed():
    """Ensures all random initializations are reproducible across test runs."""
    torch.manual_seed(42)
    if torch.cuda.is_cuda_available():
        torch.cuda.manual_seed_all(42)

@pytest.fixture
def device():
    """Dynamically routes tests to CUDA if available, otherwise falls back to CPU."""
    return torch.device("cuda" if torch.cuda.is_cuda_available() else "cpu")

@pytest.fixture
def sample_tensor(device):
    """Generates a standard mock 4D image batch tensor for activation testing."""
    # Shape: [Batch=2, Channels=16, Height=32, Width=32]
    return torch.randn(2, 16, 32, 32, device=device, requires_grad=True)