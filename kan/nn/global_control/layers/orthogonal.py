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
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(input_dim, output_dim, grid_epsilon)
        self.order = int(order)

        # (I, O, P)
        self.weights = nn.Parameter(torch.empty(input_dim, output_dim, self.order + 1))
        nn.init.xavier_uniform_(self.weights)

        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None

        self.poly_calc = OrthogonalPolynomial(
            polynomial=polynomial, order=self.order, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = torch.clamp(x, -1.0 + self.grid_epsilon, 1.0 - self.grid_epsilon)
        basis_values = self.poly_calc.calculate_basis(x)  # (B, I, P)

        B = basis_values.reshape(x.shape[0], -1).contiguous()
        W = self.weights.reshape(-1, self.output_dim).contiguous()

        output = B @ W
        if self.bias is not None:
            output = output + self.bias
        return output
