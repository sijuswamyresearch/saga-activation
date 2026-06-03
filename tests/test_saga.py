"""
tests/test_saga.py
==================
pytest test suite for the SAGA activation package.

Run with:
    pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn

from src.saga import SAGA, SAGALayer, SAGAResBlock, SAGABottleneck
from src.saga.utils import count_parameters, freeze_gate, unfreeze_gate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


@pytest.fixture(params=DEVICES)
def device(request):
    return torch.device(request.param)


# ---------------------------------------------------------------------------
# 1. Output shape tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    @pytest.mark.parametrize("B,C,H,W", [
        (2, 1, 8, 8),      # B=2 to avoid BatchNorm crash during train mode
        (2, 16, 32, 32),
        (4, 64, 128, 128),
        (2, 3, 256, 256),
    ])
    def test_shape_preserved(self, B, C, H, W):
        """SAGA must return a tensor with the same shape as the input."""
        act = SAGA(in_channels=C)
        x = torch.randn(B, C, H, W)
        y = act(x)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_single_channel(self):
        act = SAGA(in_channels=1)
        x = torch.randn(2, 1, 4, 4)
        assert act(x).shape == x.shape

    def test_large_channel(self):
        act = SAGA(in_channels=512)
        x = torch.randn(2, 512, 8, 8)
        assert act(x).shape == x.shape


# ---------------------------------------------------------------------------
# 2. Edge-case tensor value tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_input(self):
        act = SAGA(in_channels=8)
        x = torch.zeros(2, 8, 16, 16)
        y = act(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()

    def test_negative_input(self):
        act = SAGA(in_channels=8)
        x = -torch.abs(torch.randn(2, 8, 16, 16))
        y = act(x)
        assert torch.isfinite(y).all()

    def test_large_positive_input(self):
        act = SAGA(in_channels=8)
        x = torch.full((2, 8, 16, 16), 1e3)
        y = act(x)
        assert torch.isfinite(y).all()

    def test_large_negative_input(self):
        act = SAGA(in_channels=8)
        x = torch.full((2, 8, 16, 16), -1e3)
        y = act(x)
        assert torch.isfinite(y).all()

    def test_nan_free(self):
        """Confirm no NaN leaks on random inputs."""
        torch.manual_seed(42)
        act = SAGA(in_channels=32)
        for _ in range(10):
            x = torch.randn(2, 32, 64, 64) * 10
            y = act(x)
            assert not torch.isnan(y).any(), "NaN detected in SAGA output"


# ---------------------------------------------------------------------------
# 3. Gradient / backprop tests
# ---------------------------------------------------------------------------

class TestGradients:
    def test_backward_cpu(self):
        act = SAGA(in_channels=16)
        x = torch.randn(2, 16, 32, 32, requires_grad=True)
        y = act(x).sum()
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_backward_cuda(self):
        act = SAGA(in_channels=16).cuda()
        x = torch.randn(2, 16, 32, 32, requires_grad=True, device="cuda")
        y = act(x).sum()
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_gate_parameters_get_gradients(self):
        act = SAGA(in_channels=8)
        x = torch.randn(2, 8, 16, 16)
        act(x).sum().backward()
        for name, p in act.named_parameters():
            assert p.grad is not None, f"No gradient for parameter '{name}'"


# ---------------------------------------------------------------------------
# 4. Device tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    def test_forward_cuda(self):
        act = SAGA(in_channels=32).cuda()
        x = torch.randn(2, 32, 64, 64, device="cuda")
        y = act(x)
        assert y.device.type == "cuda"
        assert y.shape == x.shape

    def test_no_memory_leak(self):
        """Check GPU memory does not grow unboundedly over repeated calls."""
        act = SAGA(in_channels=64).cuda()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        for _ in range(50):
            x = torch.randn(4, 64, 128, 128, device="cuda")
            _ = act(x)
            del x

        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        # Allow at most 200 MB overhead
        assert (peak - baseline) < 200 * 1024 ** 2, "Potential memory leak detected"


# ---------------------------------------------------------------------------
# 5. Building-block tests
# ---------------------------------------------------------------------------

class TestBlocks:
    def test_resblock_shape(self):
        block = SAGAResBlock(64)
        x = torch.randn(2, 64, 32, 32)
        assert block(x).shape == x.shape

    def test_resblock_projection(self):
        block = SAGAResBlock(32, out_channels=64, stride=2)
        x = torch.randn(2, 32, 32, 32)
        y = block(x)
        assert y.shape == (2, 64, 16, 16)

    def test_bottleneck_shape(self):
        block = SAGABottleneck(64)
        x = torch.randn(2, 64, 32, 32)
        assert block(x).shape == x.shape

    def test_bottleneck_channel_change(self):
        block = SAGABottleneck(32, out_channels=128)
        x = torch.randn(2, 32, 16, 16)
        assert block(x).shape == (2, 128, 16, 16)


# ---------------------------------------------------------------------------
# 6. Utility tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_count_parameters(self):
        act = SAGA(in_channels=64)
        n = count_parameters(act)
        assert n > 0

    def test_freeze_unfreeze_gate(self):
        model = nn.Sequential(SAGA(32), SAGA(32))
        
        # Test Freeze
        freeze_gate(model)
        for module in model.modules():
            if isinstance(module, SAGA):
                for p in module.spatial_conv.parameters():
                    assert not p.requires_grad
                for p in module.gate_generator.parameters():
                    assert not p.requires_grad

        # Test Unfreeze
        unfreeze_gate(model)
        for module in model.modules():
            if isinstance(module, SAGA):
                for p in module.spatial_conv.parameters():
                    assert p.requires_grad
                for p in module.gate_generator.parameters():
                    assert p.requires_grad


# ---------------------------------------------------------------------------
# 7. SAGALayer alias
# ---------------------------------------------------------------------------

def test_saga_layer_alias():
    assert SAGALayer is SAGA


# ---------------------------------------------------------------------------
# 8. Serialisation / state-dict round-trip
# ---------------------------------------------------------------------------

def test_state_dict_round_trip(tmp_path):
    act = SAGA(in_channels=16)
    x = torch.randn(2, 16, 8, 8)
    y_before = act(x)

    path = tmp_path / "saga.pt"
    torch.save(act.state_dict(), path)

    act2 = SAGA(in_channels=16)
    act2.load_state_dict(torch.load(path, map_location="cpu"))
    y_after = act2(x)

    assert torch.allclose(y_before, y_after, atol=1e-6)