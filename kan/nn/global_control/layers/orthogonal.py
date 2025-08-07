import torch
import torch.nn as nn
from ..basis.orthogonal import OrthogonalPolynomial
from .base import GlobalControlKANLayer


class OrthogonalKANLayer(GlobalControlKANLayer):
    """KAN layer with orthogonal polynomial basis and global control."""

    def __init__(
        self,
        input_dim,
        output_dim,
        polynomial="legendre",
        order=3,
        grid_epsilon=1e-6,
        **kwargs,
    ):
        super().__init__(input_dim, output_dim, grid_epsilon)
        self.order = order

        self.weights = nn.Parameter(torch.empty(input_dim, output_dim, order + 1))
        nn.init.kaiming_uniform_(self.weights, mode="fan_in", nonlinearity="relu")

        self.poly_calc = OrthogonalPolynomial(
            polynomial=polynomial, order=order, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.clamp(x, -1.0 + self.grid_epsilon, 1.0 - self.grid_epsilon)
        basis_values = self.poly_calc.calculate_basis(x)
        output = torch.einsum("bip,iop->bo", basis_values, self.weights)
        return output
