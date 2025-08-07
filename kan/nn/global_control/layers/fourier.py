import torch
import torch.nn as nn
from ..basis.orthogonal import FourierBasis
from .base import GlobalControlKANLayer


class FourierKANLayer(GlobalControlKANLayer):
    """KAN layer with Fourier basis and global control."""

    def __init__(self, input_dim, output_dim, order=3, grid_epsilon=1e-6):
        super().__init__(input_dim, output_dim, grid_epsilon)
        self.order = order

        self.weights = nn.Parameter(torch.empty(input_dim, output_dim, 2 * order + 1))
        nn.init.kaiming_uniform_(self.weights, mode="fan_in", nonlinearity="relu")

        self.fourier_calc = FourierBasis(order=order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.clamp(x, -1.0 + self.grid_epsilon, 1.0 - self.grid_epsilon)
        basis_values = self.fourier_calc.calculate_basis(x)
        output = torch.einsum("bip,iop->bo", basis_values, self.weights)
        return output
