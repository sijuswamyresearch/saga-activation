Mathematical Background
=======================

Standard activations such as ReLU and SiLU apply the same non-linearity at
every spatial location. While efficient, this is suboptimal for complex imaging 
modalities where critical information—such as fine structural boundaries, 
tumor margins, or high-frequency foreground textures—is spatially concentrated.

SAGA formulation
----------------

To selectively route gradient flow through high-frequency structural regions, 
SAGA extracts spatial context, calculates a residual boost, and dynamically 
gates this boost before adding it back to the identity.

In version 0.2.0, the formulation includes a temperature scaling parameter to 
control the sharpness of the gate boundaries. Given an input tensor 
:math:`\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}`, the 
Spatially-Adaptive Gated Activation computes:

.. math::

   \mathbf{T}(\mathbf{X}) &= \text{BN}\left( \mathbf{W}_s \ast_3 \mathbf{X} \right) \\
   \mathbf{B}(\mathbf{X}) &= \max(0, \mathbf{T}(\mathbf{X}) - \mathbf{X}) \\
   \mathbf{G}(\mathbf{X}) &= \sigma\left( \frac{\mathbf{W}_g \ast_1 \mathbf{T}(\mathbf{X}) + b_{init}}{\tau} \right) \\
   \text{SAGA}(\mathbf{X}) &= \mathbf{X} + \left( \mathbf{G}(\mathbf{X}) \odot \mathbf{B}(\mathbf{X}) \right)

where:

- :math:`\ast_3` is a :math:`3\times 3` depthwise convolution extracting spatial context (:math:`\mathbf{W}_s`),
- :math:`\text{BN}` is standard 2D Batch Normalization,
- :math:`\max(0, \cdot)` represents the ReLU operation calculating the positive residual boost :math:`\mathbf{B}`,
- :math:`\ast_1` is a :math:`1\times 1` pointwise convolution generating the gate (:math:`\mathbf{W}_g`),
- :math:`b_{init}` is the initial gating bias, determining early-epoch connection openness,
- :math:`\tau` is the temperature parameter controlling gate sharpness,
- :math:`\sigma` is the logistic sigmoid, and
- :math:`\odot` is element-wise (Hadamard) multiplication.

Parameter count overhead
------------------------

Unlike generic parameterless activations, SAGA introduces a small number of 
parameters to achieve its spatial adaptivity. For a layer with :math:`C` 
channels, the overhead breakdown is:

- **Depthwise spatial convolution:** :math:`9C` parameters (no bias)
- **Batch Normalization:** :math:`2C` parameters (scale and shift)
- **Pointwise gate generator:** :math:`C^2 + C` parameters (weights and bias)

Total overhead: :math:`C^2 + 12C` parameters. 

For a standard intermediate feature map where :math:`C = 64`, SAGA adds exactly 
4,864 parameters. This is a highly efficient, negligible memory footprint 
compared to the structural preservation gains achieved during general image 
restoration and dense prediction tasks.