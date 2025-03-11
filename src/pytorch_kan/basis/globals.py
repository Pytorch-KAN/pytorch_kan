import torch
import torch.nn.functional as F


class OrthogonalPolynomial:
    """
    A class for calculating orthogonal polynomial basis tensors used in Kolmogorov-Arnold Networks (KAN).

    This class supports various types of orthogonal polynomials including Legendre, Chebyshev (first and second kind),
    and Gegenbauer polynomials.

    Attributes:
        polynomial (str): The type of orthogonal polynomial to use.
        order (int): The order of the polynomial.
        activation (torch.nn.Module): Activation function to be applied (if any).
        alpha (torch.Tensor): Parameter for Gegenbauer polynomials.

    Raises:
        ValueError: If an unsupported polynomial type is specified or if the order is negative.
    """

    def __init__(
        self, polynomial: str, order: int, alpha_size: int = None, beta_size: int = None
    ):
        """
        Initialize the OrthogonalPolynomial instance.

        Args:
            polynomial (str): The type of orthogonal polynomial ('legendre', 'chebyshev_first', 'chebyshev_second', or 'gegenbauer').
            order (int): The order of the polynomial.
            activation (torch.nn.Module, optional): Activation function to be applied. Defaults to None.
            alpha (torch.Tensor, optional): Parameter for Gegenbauer polynomials. Defaults to None.

        Raises:
            ValueError: If an unsupported polynomial type is specified or if the order is negative.
        """

        self.POLY_WRAPPER = {
            "legendre": self._legendre_matrix,
            "chebyshev_first": self._chebyshev_first_matrix,
            "chebyshev_second": self._chebyshev_second_matrix,
            "gegenbauer": self._gegenbauer_matrix,
            "hermite": self._hermite_matrix,
            "laguerre": self._laguerre_matrix,
            "jacobi": self._jacobi_matrix,
        }

        assert (
            polynomial in self.POLY_WRAPPER.keys()
        ), f"Unsupported polynomial type: {polynomial}"
        assert order >= 0, "Order must be a non-negative integer."

        self.polynomial = polynomial
        self.order = order

        if self.polynomial == "gegenbauer":
            assert (
                alpha_size is not None
            ), "alpha_size must be specified for Gegenbauer polynomials."
            self.activation = torch.nn.SiLU()
            self.alpha = torch.nn.Parameter(
                torch.rand(alpha_size, alpha_size), requires_grad=True
            )

        if self.polynomial == "jacobi":
            assert (
                alpha_size is not None and beta_size is not None
            ), "alpha_size and beta_size required for Jacobi"
            self.activation = torch.nn.SiLU()
            self.alpha = torch.nn.Parameter(torch.rand(alpha_size), requires_grad=True)
            self.beta = torch.nn.Parameter(torch.rand(beta_size), requires_grad=True)

    def calculate_polynomial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the orthogonal polynomial basis tensor for the given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Calculated orthogonal polynomial basis tensor.

        Raises:
            ValueError: If an unsupported polynomial type is encountered.
        """
        return self.POLY_WRAPPER[self.polynomial](x)

    def _legendre_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Legendre polynomial basis tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Legendre polynomial basis tensor.
        """

        polys = torch.ones(*x.shape, self.order + 1, device=x.device)
        if self.order > 0:
            polys[..., 1] = x
            for n in range(2, self.order + 1):
                polys[..., n] = (
                    (2 * n - 1) * x * polys[..., n - 1] - (n - 1) * polys[..., n - 2]
                ) / n
        return polys

    def _chebyshev_first_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Chebyshev polynomial (first kind) basis tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Chebyshev polynomial (first kind) basis tensor.
        """
        polys = torch.ones(*x.shape, self.order + 1, device=x.device)
        if self.order > 0:
            polys[..., 1] = x
            for n in range(2, self.order + 1):
                polys[..., n] = 2 * x * polys[..., n - 1] - polys[..., n - 2]
        return polys

    def _chebyshev_second_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Chebyshev polynomial (second kind) basis tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Chebyshev polynomial (second kind) basis tensor.
        """
        polys = torch.ones(*x.shape, self.order + 1, device=x.device)
        if self.order > 0:
            polys[..., 1] = 2 * x
            for n in range(2, self.order + 1):
                polys[..., n] = 2 * x * polys[..., n - 1] - polys[..., n - 2]
        return polys

    def _gegenbauer_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Gegenbauer polynomial basis tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Gegenbauer polynomial basis tensor.

        Raises:
            ValueError: If alpha is not set or if an activation function is provided.
        """
        constrained_alpha = self.activation(self.alpha)
        polys = torch.ones(*x.shape, self.order + 1, device=x.device)
        if self.order > 0:
            polys[..., 1] = 2 * constrained_alpha * x
            if self.order > 1:
                n_range = torch.arange(1, self.order, device=x.device)
                coeff_matrix = torch.zeros(self.order - 1, 3, device=x.device)
                coeff_matrix[:, 0] = -(n_range + 2 * constrained_alpha - 2) / (
                    n_range + 1
                )
                coeff_matrix[:, 1] = (
                    2 * (n_range + constrained_alpha - 1) / (n_range + 1)
                )
                polys[..., 2:] = torch.einsum(
                    "bik,kj->bij",
                    polys[..., 1:-1].unsqueeze(-1) * x.unsqueeze(-1),
                    coeff_matrix[:, 1:],
                ) + torch.einsum(
                    "bik,k->bi", polys[..., :-2], coeff_matrix[:, 0]
                ).unsqueeze(
                    -1
                )
        return polys

    def _hermite_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Physicists' Hermite polynomial basis"""
        polys = torch.ones(*x.shape, self.order + 1, device=x.device)
        if self.order > 0:
            polys[..., 1] = 2 * x
            for n in range(2, self.order + 1):
                polys[..., n] = (
                    2 * x * polys[..., n - 1] - 2 * (n - 1) * polys[..., n - 2]
                )
        return polys

    def _laguerre_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Laguerre polynomial basis"""
        polys = torch.ones(*x.shape, self.order + 1, device=x.device)
        if self.order > 0:
            polys[..., 1] = 1 - x
            for n in range(2, self.order + 1):
                polys[..., n] = (
                    (2 * n - 1 - x) * polys[..., n - 1] - (n - 1) * polys[..., n - 2]
                ) / n
        return polys

    def _jacobi_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Jacobi polynomial basis P^(α,β)_n(x)"""
        constrained_alpha = self.activation(self.alpha)
        constrained_beta = self.activation(self.beta)

        polys = torch.ones(*x.shape, self.order + 1, device=x.device)

        if self.order > 0:
            a, b = constrained_alpha, constrained_beta
            polys[..., 1] = 0.5 * ((a - b) + (a + b + 2) * x)

            for n in range(2, self.order + 1):
                A = 2 * n * (n + a + b) * (2 * n + a + b - 2)
                B = (2 * n + a + b - 1) * (
                    a**2 - b**2 + x * (2 * n + a + b - 2) * (2 * n + a + b)
                )
                C = 2 * (n + a - 1) * (n + b - 1) * (2 * n + a + b)

                polys[..., n] = (B * polys[..., n - 1] - C * polys[..., n - 2]) / A

        return polys
