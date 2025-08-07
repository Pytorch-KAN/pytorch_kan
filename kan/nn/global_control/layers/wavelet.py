import torch
import torch.nn as nn
from ..basis.orthogonal import WaveletBasis
from .base import GlobalControlKANLayer


class WaveletKANLayer(GlobalControlKANLayer):
    """KAN layer using wavelet basis with global control."""

    def __init__(self, input_dim, output_dim, order=3):
        super().__init__(input_dim, output_dim)
        self.order = order

        self.weights = nn.Parameter(torch.empty(input_dim, output_dim, order))
        nn.init.kaiming_uniform_(self.weights, mode="fan_in", nonlinearity="relu")

        self.wavelet_calc = WaveletBasis(order=order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        basis_values = self.wavelet_calc.calculate_basis(x)
        output = torch.einsum("bip,iop->bo", basis_values, self.weights)
        return output
