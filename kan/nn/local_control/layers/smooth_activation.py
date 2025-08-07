import torch
import torch.nn as nn
from ..basis.smooth_activation import SmoothActivationBasis
from .base import LocalControlKANLayer


class SmoothActivationKANLayer(LocalControlKANLayer):
    """KAN layer using smooth activation function basis with local control."""

    def __init__(
        self,
        input_dim,
        output_dim,
        activation_type="silu",
        alpha=1.0,
        beta=1.0,
    ):
        super().__init__(input_dim, output_dim)

        # Learnable weights (single basis function)
        self.weights = nn.Parameter(torch.empty(input_dim, output_dim, 1))
        nn.init.kaiming_uniform_(self.weights, mode="fan_in", nonlinearity="relu")

        # Basis calculator
        self.activation_calc = SmoothActivationBasis(
            order=1, activation_type=activation_type, alpha=alpha, beta=beta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        basis_values = self.activation_calc.calculate_basis(x)
        if basis_values.dim() == 2:
            basis_values = basis_values.unsqueeze(-1)
        output = torch.einsum("bip,iop->bo", basis_values, self.weights)
        return output
