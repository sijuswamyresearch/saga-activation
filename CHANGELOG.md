# Changelog

All notable changes to SAGA are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2026-06-17

### Added

- **Triton Fused Kernels:** Native support for high-speed GPU execution via `triton`, fusing pointwise mathematical operations and dual-gradient backward passes directly in SRAM.
- **Dynamic Interpretability Mode:** Added the `return_gate` parameter to the `SAGA` module to extract `(output, gate_map)` tuples for post-hoc heatmap analysis or active training-time supervision (e.g., Gate Alignment Loss).
- **Hyperparameter Controls:** Added `temperature` to adjust gate boundary sharpness and `init_bias` to control early-epoch background suppression/openness.
- **Global Toggling:** Added `set_return_gate` utility to `saga.utils` for enabling/disabling interpretability modes across entire model architectures with a single function call.

### Changed

- **Unrolled Building Blocks:** Refactored `SAGAResBlock` and `SAGABottleneck` to completely remove `nn.Sequential`. These blocks now safely intercept, unpack, and route tuple arrays when `return_gate=True` without causing PyTorch routing crashes.
- **Curriculum Learning Logic:** Updated `freeze_gate` and `unfreeze_gate` (`saga.utils`) to properly target the entire SAGA block (`spatial_conv`, `spatial_bn`, and `gate_generator`), ensuring true static identity mapping during early backbone training.
- **Test Suite Expansion:** Upgraded `pytest` coverage to explicitly validate dual-gradient flow during active gate supervision, structural tuple unpacking, and dynamic parameter (`temperature`, `init_bias`) integration.

### Performance

- Massive reduction in VRAM allocation and memory bandwidth bottlenecks by bypassing PyTorch's eager-mode intermediate tensor storage during the forward and backward passes.

---

## [0.1.4] — 2026-06-07

### Added

- Core `SAGA` operator (`saga.activation`) implementing dynamic spatial gating, 
  spatial context extraction, and an additive residual boost, matching the 
  original research formulation.
- `SAGALayer` as a direct alias for `SAGA`.
- `SAGAResBlock` and `SAGABottleneck` building blocks (`saga.blocks`).
- `count_parameters`, `freeze_gate`, `unfreeze_gate` helpers (`saga.utils`).
- Comprehensive `pytest` test suite covering shape invariance, edge cases
  (zero, negative, large inputs), gradient correctness, CUDA memory safety,
  and state-dict serialisation.
- GitHub Actions CI workflow (Ubuntu + Windows, Python 3.10–3.12).
- Sphinx documentation source with API reference and mathematical background.
- Zenodo metadata file for DOI minting.