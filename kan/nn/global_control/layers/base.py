"""Base classes for global-control KAN layers.

This module defines :class:`GlobalControlKANLayer`, the common ancestor for
layers whose basis functions have **global support**.  In such layers, modifying
one coefficient typically influences the function across the entire input
domain.  Typical bases include Fourier series, orthogonal polynomials and
wavelets.
"""

from __future__ import annotations

import torch.nn as nn


class GlobalControlKANLayer(nn.Module):
    """Base class for KAN layers with global-control basis functions.

    Parameters
    ----------
    input_dim:
        Size of the incoming features.
    output_dim:
        Size of the output features.
    grid_epsilon:
        Small constant used when clamping inputs to avoid numerical issues in
        bases defined on compact intervals.

    Notes
    -----
    Global-control layers use basis functions whose support spans the whole
    input range.  Consequently, parameter updates tend to have a global effect
    on the represented function.  Subclasses inherit a common
    :class:`~torch.nn.LayerNorm` instance for input normalisation.
    """

    def __init__(
        self, input_dim: int, output_dim: int, grid_epsilon: float = 1e-6
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_epsilon = grid_epsilon

        # Normalise inputs before computing basis expansions
        self.norm = nn.LayerNorm(input_dim)

