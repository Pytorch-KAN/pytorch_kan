import torch
import torch.nn.functional as F
from functools import lru_cache
from src.basis.base import BaseBasis

class BSplineBasis(BaseBasis):
    def __init__(self, order: int, degree: int, knots: torch.Tensor):
        super().__init__(order)
        self.degree = degree
        self.knots = knots

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        return self.bspline_basis_expansion(x, self.degree, self.knots, self.order)

    @staticmethod
    @lru_cache(maxsize=None)
    def _binomial_coefficients_matrix(spline_degree):
        basis = torch.zeros(spline_degree + 1, spline_degree + 1)
        for i in range(spline_degree + 1):
            for j in range(i + 1):
                coef = 1
                for k in range(1, j + 1):
                    coef *= (i - k + 1)
                    coef //= k
                basis[i, j] = coef * (-1)**(i - j)
        return basis

    @staticmethod
    @lru_cache(maxsize=None)
    def _vandermonde_matrix(spline_degree):
        t = torch.linspace(0, 1, spline_degree + 1)
        basis_matrix = torch.zeros((spline_degree + 1, spline_degree + 1))
        for i in range(spline_degree + 1):
            basis_matrix[i] = t ** i
        return torch.inverse(basis_matrix)

    @staticmethod
    def cox_de_boor_basis(x, i, k, knots):
        if k == 0:
            return 1.0 if knots[i] <= x < knots[i+1] else 0.0
        d1 = knots[i+k] - knots[i]
        d2 = knots[i+k+1] - knots[i+1]
        f1 = 0.0 if d1 == 0.0 else (x - knots[i]) / d1
        f2 = 0.0 if d2 == 0.0 else (knots[i+k+1] - x) / d2
        return f1 * BSplineBasis.cox_de_boor_basis(x, i, k-1, knots) + f2 * BSplineBasis.cox_de_boor_basis(x, i+1, k-1, knots)

    @staticmethod
    def _de_boor_matrix(x, knots, coefficients, degree):
        batch_size, input_dim = x.shape
        indices = torch.searchsorted(knots[degree:-degree], x.flatten()) - 1
        indices = torch.clamp(indices, 0, len(knots) - degree - 2)
        d = torch.zeros((batch_size * input_dim, degree + 1), device=x.device)
        for i in range(degree + 1):
            idx = indices + i
            d[:, i] = coefficients[idx]
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                alpha = torch.zeros_like(x.flatten())
                left_knot = knots[indices + j - r]
                right_knot = knots[indices + j]
                denominator = right_knot - left_knot
                mask = denominator != 0
                alpha[mask] = (x.flatten()[mask] - left_knot[mask]) / denominator[mask]
                d[:, j] = (1 - alpha) * d[:, j-1] + alpha * d[:, j]
        return d[:, degree].reshape(batch_size, input_dim)

    @staticmethod
    def bspline_basis_expansion(x, degree, knots, num_bases):
        batch_size, input_dim = x.shape
        basis_values = torch.zeros(batch_size, input_dim, num_bases, device=x.device)
        for i in range(num_bases):
            for b in range(batch_size):
                for d in range(input_dim):
                    basis_values[b, d, i] = BSplineBasis.cox_de_boor_basis(x[b, d], i, degree, knots)
        return basis_values

class RBFBasis(BaseBasis):
    def __init__(self, order: int, centers: torch.Tensor, sigma: float):
        super().__init__(order)
        self.centers = centers
        self.sigma = sigma

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        return self.rbf_basis(x, self.centers, self.sigma)

    @staticmethod
    def rbf_basis(x, centers, sigma):
        batch_size, input_dim = x.shape
        num_rbf_bases = len(centers)
        rbf_values = torch.zeros(batch_size, input_dim, num_rbf_bases, device=x.device)
        for i, center in enumerate(centers):
            squared_dist = torch.sum((x.unsqueeze(2) - center.view(1, 1, -1))**2, dim=-1)
            rbf_values[:, :, i] = torch.exp(-squared_dist / (sigma**2))
        return rbf_values

class SmoothActivationBasis(BaseBasis):
    def __init__(self, order: int, activation_type: str, alpha: float = 1.0, beta: float = 1.0):
        super().__init__(order)
        self.activation_type = activation_type
        self.alpha = alpha
        self.beta = beta

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == 'silu':
            return self._silu(x)
        elif self.activation_type == 'gelu':
            return self._gelu(x)
        elif self.activation_type == 'softplus':
            return self._softplus(x)
        elif self.activation_type == 'mish':
            return self._mish(x)
        elif self.activation_type == 'elu':
            return self._elu(x)
        elif self.activation_type == 'tanh_shrink':
            return self._tanh_shrink(x)
        else:
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


