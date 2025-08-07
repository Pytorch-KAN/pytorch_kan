"""Base classes for local-control KAN layers.

This module defines the :class:`LocalControlKANLayer` base class used by all
layers that employ basis functions with **local support**.  In these layers a
change to one coefficient influences only a neighbourhood of the input domain,
providing *local control* over the function representation.  Examples include
layers built from B-splines or radial basis functions.
"""

from __future__ import annotations

import torch.nn as nn


class LocalControlKANLayer(nn.Module):
    """Base class for KAN layers with local-control basis functions.

    Parameters
    ----------
    input_dim:
        Size of the incoming features.
    output_dim:
        Size of the output features.

    Notes
    -----
    Local-control layers utilise basis functions with compact support.  This
    means adjustments to individual coefficients affect only a small region of
    the input space.  Subclasses inherit a common :class:`~torch.nn.LayerNorm`
    module for input normalisation and store ``input_dim``/``output_dim``
    attributes.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Normalise inputs before computing basis expansions
        self.norm = nn.LayerNorm(input_dim)

