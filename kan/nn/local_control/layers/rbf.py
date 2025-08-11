import torch
import torch.nn as nn
from ..basis.rbf import RBFBasis
from .base import LocalControlKANLayer


class RBFKANLayer(LocalControlKANLayer):
    """KAN layer with per-dimension Gaussian RBFs and local control."""

    def __init__(self, input_dim, output_dim, num_centers=10, sigma=1.0, bias=True):
        super().__init__(input_dim, output_dim)
        self.num_centers = int(num_centers)

        # Per-dimension centres: (input_dim, num_centers)
        self.centers = nn.Parameter(torch.randn(input_dim, num_centers))
        # Raw sigma parameter, constrained to >0 via softplus in ``sigma`` property
        self._sigma_raw = nn.Parameter(torch.tensor(float(sigma)))

        # Weights mixing the basis functions
        self.weights = nn.Parameter(torch.empty(input_dim, output_dim, num_centers))
        nn.init.xavier_uniform_(self.weights)

        self.bias = nn.Parameter(torch.zeros(output_dim)) if bias else None

        # Basis calculator
        self.rbf_calc = RBFBasis(order=num_centers, centers=self.centers)

    @property
    def sigma(self):
        """Positive width parameter for the RBFs."""

        return torch.nn.functional.softplus(self._sigma_raw) + 1e-6

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)  # RBFs do not require clamping to [-1, 1]

        basis = self.rbf_calc.calculate_basis(x, sigma=self.sigma)  # (B, I, C)
        # (B, I, C) @ (I, C, O) -> (B, I, O)
        per_dim = torch.matmul(basis, self.weights.transpose(-1, -2))
        out = per_dim.sum(dim=1)  # (B, O)

        if self.bias is not None:
            out = out + self.bias
        return out

