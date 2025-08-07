import torch
from functools import lru_cache
from pydantic import BaseModel, validator

from ...basis.base import BaseBasis
from ....utils import logger, progress, status


class BSplineBasisParams(BaseModel):
    order: int
    degree: int
    knots: torch.Tensor

    @validator("order", "degree")
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("order and degree must be non-negative")
        return v

    @validator("knots")
    def tensor_knots(cls, v):
        if not isinstance(v, torch.Tensor):
            raise TypeError("knots must be a torch.Tensor")
        return v

    class Config:
        arbitrary_types_allowed = True


class BSplineBasis(BaseBasis):
    """B-spline basis functions for local control."""

    def __init__(self, order: int, degree: int, knots: torch.Tensor):
        """Parameters
        ----------
        order: int
            Number of spline basis functions.
        degree: int
            Polynomial degree of the splines.
        knots: torch.Tensor
            Knot vector defining the spline segments.
        """
        params = BSplineBasisParams(order=order, degree=degree, knots=knots)
        super().__init__(params.order)
        self.degree = params.degree
        self.knots = params.knots
        logger.debug(
            f"Initialized BSplineBasis with order={self.order}, degree={self.degree}, knots_shape={self.knots.shape}"
        )

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis for ``x``."""
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
                basis[i, j] = coef * (-1) ** (i - j)
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
            return 1.0 if knots[i] <= x < knots[i + 1] else 0.0
        d1 = knots[i + k] - knots[i]
        d2 = knots[i + k + 1] - knots[i + 1]
        f1 = 0.0 if d1 == 0.0 else (x - knots[i]) / d1
        f2 = 0.0 if d2 == 0.0 else (knots[i + k + 1] - x) / d2
        return f1 * BSplineBasis.cox_de_boor_basis(x, i, k - 1, knots) + f2 * BSplineBasis.cox_de_boor_basis(x, i + 1, k - 1, knots)

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
                d[:, j] = (1 - alpha) * d[:, j - 1] + alpha * d[:, j]
        return d[:, degree].reshape(batch_size, input_dim)

    @staticmethod
    def bspline_basis_expansion(x, degree, knots, num_bases):
        batch_size, input_dim = x.shape
        basis_values = torch.zeros(batch_size, input_dim, num_bases, device=x.device)
        with status("Evaluating B-spline basis"):
            for i in progress(range(num_bases), "Basis functions"):
                for b in range(batch_size):
                    for d in range(input_dim):
                        basis_values[b, d, i] = BSplineBasis.cox_de_boor_basis(
                            x[b, d], i, degree, knots
                        )
        return basis_values
