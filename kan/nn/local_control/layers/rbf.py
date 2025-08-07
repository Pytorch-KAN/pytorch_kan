import torch
import torch.nn as nn
from ..basis.rbf import RBFBasis
from .base import LocalControlKANLayer


class RBFKANLayer(LocalControlKANLayer):
    """KAN layer with radial basis functions and local control."""

    def __init__(self, input_dim, output_dim, num_centers=10, sigma=1.0):
        super().__init__(input_dim, output_dim)
        self.num_centers = num_centers
        self.sigma = sigma

        # Learnable RBF centers
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))

        # Learnable weights for combining RBF outputs
        self.weights = nn.Parameter(torch.empty(input_dim, output_dim, num_centers))
        nn.init.kaiming_uniform_(self.weights, mode="fan_in", nonlinearity="relu")

        # Basis calculator
        self.rbf_calc = RBFBasis(order=num_centers, centers=self.centers, sigma=sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        basis_values = self.rbf_calc.calculate_basis(x)
        output = torch.einsum("bip,iop->bo", basis_values, self.weights)
        return output
