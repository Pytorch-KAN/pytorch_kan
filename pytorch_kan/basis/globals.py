import torch
import torch.nn.functional as F
from src.pytorch_kan.basis.base import BaseBasis


class OrthogonalPolynomial(BaseBasis):
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


class FourierBasis(BaseBasis):
    """
    A class for calculating Fourier basis tensors used in Kolmogorov-Arnold Networks (KAN).

    Attributes:
        order (int): The order of the Fourier basis.
    
    Raises:
        ValueError: If the order is negative.
    """

    def __init__(self, order: int):
        """
        Initialize the FourierBasis instance.

        Args:
            order (int): The order of the Fourier basis.

        Raises:
            ValueError: If the order is negative.
        """
        assert order >= 0, "Order must be a non-negative integer."

        self.order = order

    def calculate_fourier(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Fourier basis tensor for the given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Calculated Fourier basis tensor with both sine and cosine components.
        """
        # Scale input to [0, 2π] for proper periodicity
        x_scaled = torch.clamp((x + 1), -1, 1) * torch.pi  # Map from [-1, 1] to [-π, π]
        
        # Initialize result tensor with 2*order+1 basis functions:
        # - 1 constant function (DC component)
        # - order sine functions
        # - order cosine functions
        result = torch.zeros(*x.shape, 2 * self.order + 1, device=x.device)
        
        # DC component (constant)
        result[..., 0] = 1.0
        
        # Generate frequency components
        for k in range(1, self.order + 1):
            # Cosine components: cos(kπx)
            result[..., k] = torch.cos(k * x_scaled)
            
            # Sine components: sin(kπx)
            result[..., k + self.order] = torch.sin(k * x_scaled)
        
        return result


class SmoothActivationBasis(BaseBasis):
    """
    A class for calculating basis tensors using smooth activation functions in Kolmogorov-Arnold Networks (KAN).

    This class supports various types of smooth activation functions including:
    - SiLU/Swish: x * sigmoid(x)
    - GELU: x * Φ(x) where Φ is the CDF of the standard normal distribution
    - Softplus: log(1 + exp(x))
    - Mish: x * tanh(softplus(x))
    - ELU: x if x > 0 else α * (exp(x) - 1)
    - TanhShrink: x - tanh(x)

    Attributes:
        activation_type (str): The type of smooth activation function to use.
        alpha (float): Parameter for certain activation functions like ELU.
        beta (float): Additional parameter for certain activation functions.
        
    Raises:
        ValueError: If an unsupported activation type is specified.
    """

    def __init__(self, activation_type: str, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize the SmoothActivationBasis instance.

        Args:
            activation_type (str): The type of smooth activation function
                                  ('silu', 'gelu', 'softplus', 'mish', 'elu', 'tanh_shrink').
            alpha (float, optional): Parameter for certain activation functions. Defaults to 1.0.
            beta (float, optional): Additional parameter for certain activation functions. Defaults to 1.0.

        Raises:
            ValueError: If an unsupported activation type is specified.
        """
        self.ACTIVATION_WRAPPER = {
            "silu": self._silu,
            "gelu": self._gelu,
            "softplus": self._softplus,
            "mish": self._mish,
            "elu": self._elu,
            "tanh_shrink": self._tanh_shrink,
        }
        
        assert activation_type in self.ACTIVATION_WRAPPER.keys(), f"Unsupported activation type: {activation_type}"
        
        self.activation_type = activation_type
        self.alpha = alpha
        self.beta = beta
        
    def calculate_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the basis tensor using the specified smooth activation function for the given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Calculated basis tensor using the specified activation function.
        """
        return self.ACTIVATION_WRAPPER[self.activation_type](x)
    
    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU/Swish activation function: x * sigmoid(x)
        """
        return x * torch.sigmoid(x)
    
    def _gelu(self, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation function: x * Φ(x) where Φ is the CDF of the standard normal distribution
        This uses the approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        """
        return F.gelu(x)
    
    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        """
        Softplus activation function: log(1 + exp(β*x))/β
        Uses the beta parameter for scaling the sharpness
        """
        # Stable implementation to avoid overflow
        result = torch.zeros_like(x)
        mask = x > 20  # threshold to avoid overflow in exp
        
        # For large values, softplus(x) ≈ x
        result[mask] = x[mask]
        
        # For small/medium values, use the standard formula
        not_mask = ~mask
        result[not_mask] = torch.log(1 + torch.exp(self.beta * x[not_mask])) / self.beta
        
        return result
    
    def _mish(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mish activation function: x * tanh(softplus(x))
        """
        return x * torch.tanh(F.softplus(x))
    
    def _elu(self, x: torch.Tensor) -> torch.Tensor:
        """
        ELU activation function: x if x > 0 else α * (exp(x) - 1)
        Uses the alpha parameter to control the negative slope
        """
        return F.elu(x, alpha=self.alpha)
    
    def _tanh_shrink(self, x: torch.Tensor) -> torch.Tensor:
        """
        TanhShrink activation function: x - tanh(x)
        """
        return x - torch.tanh(x)


class WaveletBasis(BaseBasis):
    def __init__(self, order: int):
        self.order = order

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim = x.shape
        wavelet_basis = torch.zeros(batch_size, input_dim, self.order, device=x.device)
        for i in range(self.order):
            wavelet_basis[..., i] = self.haar_wavelet(x, i)
        return wavelet_basis

    def haar_wavelet(self, x: torch.Tensor, order: int) -> torch.Tensor:
        if order == 0:
            return torch.ones_like(x)
        else:
            k = 2 ** (order - 1)
            return torch.where((x >= 0) & (x < 1 / k), 1, -1)


class FourierBasis(BaseBasis):
    def __init__(self, order: int):
        self.order = order

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        x_scaled = torch.clamp((x + 1), -1, 1) * torch.pi
        result = torch.zeros(*x.shape, 2 * self.order + 1, device=x.device)
        result[..., 0] = 1.0
        for k in range(1, self.order + 1):
            result[..., k] = torch.cos(k * x_scaled)
            result[..., k + self.order] = torch.sin(k * x_scaled)
        return result