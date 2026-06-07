# Changelog

All notable changes to SAGA are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.3] — 2026-06-03

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