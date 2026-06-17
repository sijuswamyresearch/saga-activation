"""
tests/test_saga.py
==================
pytest test suite for the SAGA activation package v0.2.0.

Run with:
    pytest tests/ -v
"""

import pytest
import torch
import torch.nn as nn

# Package is imported natively assuming `pip install -e .`
from saga import SAGA, SAGALayer, SAGAResBlock, SAGABottleneck
from saga.utils import count_parameters, freeze_gate, unfreeze_gate, set_return_gate

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
# 1. Output shape & Interpretability tests
# ---------------------------------------------------------------------------

class TestOutputShape:
    @pytest.mark.parametrize("B,C,H,W", [
        (2, 1, 8, 8),      # B=2 to avoid BatchNorm crash during train mode
        (2, 16, 32, 32),
        (4, 64, 128, 128),
    ])
    def test_shape_preserved(self, B, C, H, W):
        """SAGA must return a single tensor by default to match native activations."""
        act = SAGA(in_channels=C, return_gate=False)
        x = torch.randn(B, C, H, W)
        y = act(x)
        assert isinstance(y, torch.Tensor)
        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_interpretability_mode(self):
        """Verifies SAGA returns a tuple (out, gate) when return_gate=True."""
        act = SAGA(in_channels=16, return_gate=True)
        x = torch.randn(2, 16, 32, 32)
        out, gate = act(x)
        assert isinstance(out, torch.Tensor)
        assert isinstance(gate, torch.Tensor)
        assert out.shape == x.shape
        assert gate.shape == x.shape
        # Gates must strictly fall within sigmoid boundary ranges
        assert torch.all(gate >= 0.0) and torch.all(gate <= 1.0)


# ---------------------------------------------------------------------------
# 2. Edge-case tensor value tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_input(self):
        act = SAGA(in_channels=8)
        x = torch.zeros(2, 8, 16, 16)
        y = act(x)
        assert torch.isfinite(y).all()

    def test_negative_input(self):
        act = SAGA(in_channels=8)
        x = -torch.abs(torch.randn(2, 8, 16, 16))
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
    def test_backward_standard(self):
        """Standard backward pass from a single task loss."""
        act = SAGA(in_channels=16, return_gate=False)
        x = torch.randn(2, 16, 32, 32, requires_grad=True)
        y = act(x).sum()
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_dual_gradient_flow(self):
        """
        Crucial Test: Verifies gradients flow back simultaneously from 
        both the output tensor and the gate tensor during active gate supervision.
        """
        act = SAGA(in_channels=16, return_gate=True)
        x = torch.randn(2, 16, 32, 32, requires_grad=True)
        
        out, gate = act(x)
        task_loss = out.pow(2).sum()
        gate_alignment_loss = gate.pow(2).sum()
        
        # Fused backward pass
        total_loss = task_loss + gate_alignment_loss
        total_loss.backward()
        
        assert x.grad is not None
        assert act.gate_generator.weight.grad is not None
        assert act.spatial_conv.weight.grad is not None


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
    def test_resblock_standard(self):
        block = SAGAResBlock(64)
        x = torch.randn(2, 64, 32, 32)
        assert block(x).shape == x.shape

    def test_resblock_interpretability_routing(self):
        """Ensures the unrolled block safely catches tuples and returns gate lists."""
        block = SAGAResBlock(32)
        set_return_gate(block, True)
        x = torch.randn(2, 32, 32, 32)
        
        out, gates = block(x)
        assert isinstance(out, torch.Tensor)
        assert isinstance(gates, list)
        assert len(gates) == 2  # ResBlock contains two SAGA modules

    def test_bottleneck_interpretability_routing(self):
        block = SAGABottleneck(32, out_channels=64)
        set_return_gate(block, True)
        x = torch.randn(2, 32, 16, 16)
        
        out, gates = block(x)
        assert out.shape == (2, 64, 16, 16)
        assert len(gates) == 3  # Bottleneck contains three SAGA modules


# ---------------------------------------------------------------------------
# 6. Utility tests & Hyperparameters
# ---------------------------------------------------------------------------

class TestUtilsAndParams:
    def test_freeze_unfreeze_gate(self):
        model = nn.Sequential(SAGA(32), SAGA(32))
        
        # Test Freeze (Should freeze entire pathway, not just generator)
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
                for p in module.gate_generator.parameters():
                    assert p.requires_grad

    @pytest.mark.parametrize("temperature,init_bias", [(0.5, -1.0), (2.0, 3.0)])
    def test_parameter_modifications(self, temperature, init_bias):
        """Ensures hyperparameter values are registered correctly and modify the module."""
        act = SAGA(in_channels=16, temperature=temperature, init_bias=init_bias)
        assert act.temperature == temperature
        assert act.gate_generator.bias[0].item() == pytest.approx(init_bias)


# ---------------------------------------------------------------------------
# 7. Serialisation / state-dict round-trip
# ---------------------------------------------------------------------------

def test_state_dict_round_trip(tmp_path):
    act = SAGA(in_channels=16, temperature=1.5, init_bias=-1.0)
    x = torch.randn(2, 16, 8, 8)
    y_before = act(x)

    path = tmp_path / "saga.pt"
    torch.save(act.state_dict(), path)

    act2 = SAGA(in_channels=16, temperature=1.5, init_bias=-1.0)
    act2.load_state_dict(torch.load(path, map_location="cpu"))
    y_after = act2(x)

    assert torch.allclose(y_before, y_after, atol=1e-6)

def test_saga_layer_alias():
    assert SAGALayer is SAGA