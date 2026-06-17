"""
saga.activation
===============
Spatially-Adaptive Gated Activation (SAGA) operator.
Optimized with OpenAI Triton for Fused GPU Execution.
Supports standard drop-in usage, post-hoc interpretability, and active gate training.

Reference
---------
Siju K.S., Venugopal V., Kar M.K., Anandakrishnan J.
"An interpretable deep learning method for medical image deblurring and
restoration." Healthcare Analytics 9 (2026) 100468.
https://doi.org/10.1016/j.health.2026.100468
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

__all__ = ["SAGA"]

# =========================================================================
# Triton Fused Kernels
# =========================================================================
if HAS_TRITON:
    @triton.jit
    def _saga_forward_kernel(
        x_ptr, tx_ptr, gx_ptr, out_ptr, gate_ptr,
        temperature, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        tx = tl.load(tx_ptr + offsets, mask=mask)
        gx = tl.load(gx_ptr + offsets, mask=mask)

        diff = tx - x
        boost = tl.maximum(diff, 0.0)
        gate = tl.sigmoid(gx / temperature)
        out = x + gate * boost

        tl.store(out_ptr + offsets, out, mask=mask)
        tl.store(gate_ptr + offsets, gate, mask=mask)

    @triton.jit
    def _saga_backward_kernel(
        dout_ptr, dgate_ext_ptr,
        x_ptr, tx_ptr, gx_ptr,
        dx_ptr, dtx_ptr, dgx_ptr,
        temperature, n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        dout = tl.load(dout_ptr + offsets, mask=mask)
        dgate_ext = tl.load(dgate_ext_ptr + offsets, mask=mask)
        x = tl.load(x_ptr + offsets, mask=mask)
        tx = tl.load(tx_ptr + offsets, mask=mask)
        gx = tl.load(gx_ptr + offsets, mask=mask)

        diff = tx - x
        relu_mask = diff > 0.0
        boost = tl.where(relu_mask, diff, 0.0) 
        gate = tl.sigmoid(gx / temperature)                  

        # Core Mathematical Fusion: Combine internal task gradient and external gate alignment gradient
        dgate_total = (dout * boost) + dgate_ext

        dgx = dgate_total * gate * (1.0 - gate) * (1.0 / temperature)
        dtx = dout * gate * tl.where(relu_mask, 1.0, 0.0)
        dx = dout * (1.0 - gate * tl.where(relu_mask, 1.0, 0.0))

        tl.store(dx_ptr + offsets, dx, mask=mask)
        tl.store(dtx_ptr + offsets, dtx, mask=mask)
        tl.store(dgx_ptr + offsets, dgx, mask=mask)


# =========================================================================
# PyTorch Autograd Function (Tracks both outputs internally)
# =========================================================================
class SAGAFusedFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, tx, gx, temperature):
        ctx.temperature = temperature
        
        if not HAS_TRITON or not x.is_cuda:
            ctx.eager_mode = True
            boost = F.relu(tx - x)
            gate = torch.sigmoid(gx / temperature)
            out = x + (gate * boost)
            ctx.save_for_backward(x, tx, gx)
            return out, gate

        ctx.eager_mode = False
        x, tx, gx = x.contiguous(), tx.contiguous(), gx.contiguous()
        out = torch.empty_like(x)
        gate = torch.empty_like(x)
        n_elements = x.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        _saga_forward_kernel[grid](x, tx, gx, out, gate, temperature, n_elements, BLOCK_SIZE=1024)

        ctx.save_for_backward(x, tx, gx)
        return out, gate

    @staticmethod
    def backward(ctx, dout, dgate_ext):
        x, tx, gx = ctx.saved_tensors
        temperature = ctx.temperature

        # Handle instances where external gate alignment loss gradients are not provided
        if dgate_ext is None:
            dgate_ext = torch.zeros_like(dout)

        if ctx.eager_mode:
            with torch.enable_grad():
                x.requires_grad_(True)
                tx.requires_grad_(True)
                gx.requires_grad_(True)
                boost = F.relu(tx - x)
                gate = torch.sigmoid(gx / temperature)
                out = x + (gate * boost)
                torch.autograd.backward([out, gate], [dout, dgate_ext])
            return x.grad, tx.grad, gx.grad, None

        dout, dgate_ext = dout.contiguous(), dgate_ext.contiguous()
        dx, dtx, dgx = torch.empty_like(x), torch.empty_like(tx), torch.empty_like(gx)
        n_elements = x.numel()

        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        _saga_backward_kernel[grid](
            dout, dgate_ext, x, tx, gx, dx, dtx, dgx, 
            temperature, n_elements, BLOCK_SIZE=1024
        )

        return dx, dtx, dgx, None


# =========================================================================
# User-Facing Module Wrapper
# =========================================================================
class SAGA(nn.Module):
    """
    Spatially-Adaptive Gated Activation (SAGA).
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    return_gate : bool (default: False)
        If True, forward returns a tuple of (output, gate_map).
        If False, returns only the activated output tensor.
    temperature : float (default: 1.0)
        Controls sharpness scaling within the gating function.
    init_bias : float (default: 2.0)
        Initial bias for the gating module. Use lower values (e.g., -1.0)
        for early background suppression or higher values (e.g., 2.0) for 
        an mostly open connection at start.
    """
    def __init__(self, in_channels: int, return_gate: bool = False, temperature: float = 1.0, init_bias: float = 2.0) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.return_gate = return_gate
        self.temperature = temperature
        self.init_bias = init_bias
        
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
        self.spatial_bn = nn.BatchNorm2d(in_channels)
        self.gate_generator = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.spatial_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.spatial_bn.weight, 1)
        nn.init.constant_(self.spatial_bn.bias, 0)
        nn.init.constant_(self.gate_generator.weight, 0)
        nn.init.constant_(self.gate_generator.bias, self.init_bias)

    def forward(self, x: torch.Tensor):
        T_x = self.spatial_bn(self.spatial_conv(x))
        G_x = self.gate_generator(T_x)
        
        # Internal autograd always handles dual gradient calculations safely
        out, gate = SAGAFusedFunction.apply(x, T_x, G_x, self.temperature)
        
        if self.return_gate:
            return out, gate
            
        return out

    def extra_repr(self) -> str:
        return f"in_channels={self.in_channels}, temp={self.temperature}, init_bias={self.init_bias}, fused={HAS_TRITON}"

# Alias for drop-in use inside sequential blocks
SAGALayer = SAGA