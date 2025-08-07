import torch
import torch.nn.functional as F
from ...basis.base import BaseBasis


class SmoothActivationBasis(BaseBasis):
    """Smooth activation functions as basis for local control."""

    def __init__(self, order: int, activation_type: str, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(order)
        self.activation_type = activation_type
        self.alpha = alpha
        self.beta = beta

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "silu":
            return self._silu(x)
        if self.activation_type == "gelu":
            return self._gelu(x)
        if self.activation_type == "softplus":
            return self._softplus(x)
        if self.activation_type == "mish":
            return self._mish(x)
        if self.activation_type == "elu":
            return self._elu(x)
        if self.activation_type == "tanh_shrink":
            return self._tanh_shrink(x)
        raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def _gelu(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(x)
        mask = x > 20
        result[mask] = x[mask]
        not_mask = ~mask
        result[not_mask] = torch.log(1 + torch.exp(self.beta * x[not_mask])) / self.beta
        return result

    def _mish(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

    def _elu(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha)

    def _tanh_shrink(self, x: torch.Tensor) -> torch.Tensor:
        return x - torch.tanh(x)
