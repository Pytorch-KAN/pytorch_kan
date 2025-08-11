import torch
import torch.nn as nn
from ..basis.bspline import BSplineBasis
from .base import LocalControlKANLayer


class MatrixKANLayer(LocalControlKANLayer):
    """KAN layer with B-spline local control (per-dimension, per-cell polynomials)."""

    def __init__(
        self,
        input_dim,
        output_dim,
        spline_degree=3,
        grid_size=100,
        grid_epsilon=1e-6,
        bias=True,
    ):
        super().__init__(input_dim, output_dim)
        self.spline_degree = int(spline_degree)
        self.grid_size = int(grid_size)
        self.grid_epsilon = float(grid_epsilon)

        # (P, P) transform; register as non-persistent buffer so it follows device/dtype
        basis = BSplineBasis._binomial_coefficients_matrix(self.spline_degree)
        self.register_buffer("basis_matrix", basis, persistent=False)

        # Per cell coefficients: (I, O, G, P)
        self.poly_matrix = nn.Parameter(
            torch.zeros(input_dim, output_dim, grid_size, self.spline_degree + 1)
        )
        nn.init.xavier_uniform_(self.poly_matrix)

        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None

    def forward(self, x):
        B = x.size(0)
        I = self.input_dim
        O = self.output_dim
        P = self.spline_degree + 1

        # Normalise and map to [0, 1)
        x = self.norm(x)
        x = torch.clamp(x, -1.0 + self.grid_epsilon, 1.0 - self.grid_epsilon)
        x01 = (x + 1) * 0.5

        # Cell indices and local coordinates
        gx = x01 * self.grid_size
        indices = torch.clamp(gx.floor().to(torch.int64), 0, self.grid_size - 1)  # (B, I)
        t = gx - indices  # (B, I) in [0,1)

        # Powers [1, t, t^2, ..., t^(P-1)] via cumulative multiplication
        if P > 1:
            t_stack = t.unsqueeze(-1).expand(B, I, P - 1)
            powers_rest = torch.cumprod(t_stack, dim=-1)
            power = torch.cat(
                [torch.ones(B, I, 1, device=x.device, dtype=x.dtype), powers_rest],
                dim=-1,
            )
        else:
            power = torch.ones(B, I, 1, device=x.device, dtype=x.dtype)

        # Local basis via (B, I, P) @ (P, P)
        basis_values = power @ self.basis_matrix  # (B, I, P)

        # Gather coefficients per (B, I) cell
        pm = self.poly_matrix.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, I, O, G, P)
        idx = indices.unsqueeze(2).unsqueeze(-1).expand(B, I, O, P)  # (B, I, O, P)
        coeffs = torch.take_along_dim(pm, idx.unsqueeze(3), dim=3).squeeze(3)  # (B, I, O, P)

        # Batched matmul: (B*I, 1, P) @ (B*I, P, O) -> (B, I, O)
        lhs = basis_values.reshape(B * I, 1, P)
        rhs = coeffs.permute(0, 1, 3, 2).reshape(B * I, P, O)
        per_dim = torch.bmm(lhs, rhs).squeeze(1).reshape(B, I, O)

        out = per_dim.sum(dim=1)  # (B, O)
        if self.bias is not None:
            out = out + self.bias
        return out

