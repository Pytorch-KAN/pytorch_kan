"""Orthogonal and other global-control basis functions for KAN."""

import torch
import torch.nn.functional as F
from pydantic import BaseModel, root_validator, validator

from ...basis.base import BaseBasis
from ....utils import logger, status


class OrthogonalPolynomialParams(BaseModel):
    polynomial: str
    order: int
    alpha_size: int | None = None
    beta_size: int | None = None

    @validator("polynomial")
    def valid_polynomial(cls, v: str) -> str:
        allowed = {
            "legendre",
            "chebyshev_first",
            "chebyshev_second",
            "gegenbauer",
            "hermite",
            "laguerre",
            "jacobi",
        }
        if v not in allowed:
            raise ValueError(f"Unsupported polynomial type: {v}")
        return v

    @validator("order")
    def non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Order must be a non-negative integer.")
        return v

    @root_validator
    def check_sizes(cls, values):
        poly = values.get("polynomial")
        alpha = values.get("alpha_size")
        beta = values.get("beta_size")
        if poly == "gegenbauer" and alpha is None:
            raise ValueError("alpha_size must be specified for Gegenbauer polynomials.")
        if poly == "jacobi" and (alpha is None or beta is None):
            raise ValueError("alpha_size and beta_size required for Jacobi")
        return values

    class Config:
        arbitrary_types_allowed = True


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
        self, polynomial: str, order: int, alpha_size: int = None, beta_size: int = None, device=None
    ):
        """Initialize the :class:`OrthogonalPolynomial` instance.

        Parameters
        ----------
        polynomial: str
            Type of polynomial to use.  Supported values are ``'legendre'``,
            ``'chebyshev_first'``, ``'chebyshev_second'``, ``'gegenbauer'``,
            ``'hermite'``, ``'laguerre'`` and ``'jacobi'``.
        order: int
            Order of the polynomial basis.
        alpha_size: int, optional
            Size of the ``alpha`` parameter vector.  Required for Gegenbauer and
            Jacobi polynomials.
        beta_size: int, optional
            Size of the ``beta`` parameter vector.  Required for Jacobi
            polynomials.
        device: torch.device, optional
            Device on which internal parameters should be allocated.
        """
        params = OrthogonalPolynomialParams(
            polynomial=polynomial, order=order, alpha_size=alpha_size, beta_size=beta_size
        )
        device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        super().__init__(params.order, device=device)
        self.POLY_WRAPPER = {
            "legendre": self._legendre_matrix,
            "chebyshev_first": self._chebyshev_first_matrix,
            "chebyshev_second": self._chebyshev_second_matrix,
            "gegenbauer": self._gegenbauer_matrix,
            "hermite": self._hermite_matrix,
            "laguerre": self._laguerre_matrix,
            "jacobi": self._jacobi_matrix,
        }
        self.polynomial = params.polynomial
        logger.debug(
            f"Initialized OrthogonalPolynomial with polynomial={self.polynomial}, order={self.order}"
        )

        if self.polynomial == "gegenbauer":
            self.activation = torch.nn.SiLU()
            self.alpha = torch.nn.Parameter(
                torch.rand(params.alpha_size, device=self.device), requires_grad=True
            )

        if self.polynomial == "jacobi":
            self.activation = torch.nn.SiLU()
            self.alpha = torch.nn.Parameter(
                torch.rand(params.alpha_size, device=self.device), requires_grad=True
            )
            self.beta = torch.nn.Parameter(
                torch.rand(params.beta_size, device=self.device), requires_grad=True
            )

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the orthogonal polynomial basis tensor for ``x``.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape ``(batch, features)``.

        Returns
        -------
        torch.Tensor
            Basis tensor with an additional dimension containing ``order``
            polynomials.

        Raises
        ------
        ValueError
            If the dimensions of ``alpha`` or ``beta`` parameters do not match
            the feature dimension of ``x``.
        """
        if hasattr(self, "alpha") and self.alpha.device != x.device or (
            hasattr(self, "beta") and self.beta.device != x.device
        ):
            self.to(x.device)

        feature_dim = x.shape[-1]
        if self.polynomial == "gegenbauer":
            if feature_dim != self.alpha.shape[0]:
                raise ValueError(
                    f"Gegenbauer alpha size ({self.alpha.shape[0]}) does not match input feature dimension ({feature_dim})."
                )
        if self.polynomial == "jacobi":
            if feature_dim != self.alpha.shape[0] or feature_dim != self.beta.shape[0]:
                raise ValueError(
                    "Jacobi alpha/beta sizes do not match input feature dimension: "
                    f"alpha {self.alpha.shape[0]}, beta {self.beta.shape[0]}, feature {feature_dim}."
                )

        with status(f"Computing {self.polynomial} basis"):
            return self.POLY_WRAPPER[self.polynomial](x)

    def _legendre_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Legendre polynomial basis tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Legendre polynomial basis tensor.
        """
        # Create first two polynomials: P_0(x) = 1
        p0 = torch.ones_like(x)
        polys = [p0]
        
        if self.order > 0:
            p1 = x.clone()  # P_1(x) = x
            polys.append(p1)
            
            # Recursively compute higher-order polynomials
            for n in range(2, self.order + 1):
                coeff1 = (2 * n - 1) / n
                coeff2 = (n - 1) / n
                pn = coeff1 * x * polys[n-1] - coeff2 * polys[n-2]
                polys.append(pn)
        
        # Stack along a new dimension
        return torch.stack(polys, dim=-1)

    def _chebyshev_first_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Chebyshev polynomial (first kind) basis tensor.
        """
        # Create first two polynomials: T_0(x) = 1
        p0 = torch.ones_like(x)
        polys = [p0]
        
        if self.order > 0:
            p1 = x.clone()  # T_1(x) = x
            polys.append(p1)
            
            # Recursively compute higher-order polynomials
            for n in range(2, self.order + 1):
                pn = 2 * x * polys[n-1] - polys[n-2]
                polys.append(pn)
        
        return torch.stack(polys, dim=-1)

    def _chebyshev_second_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Chebyshev polynomial (second kind) basis tensor.
        """
        # Create first two polynomials: U_0(x) = 1
        p0 = torch.ones_like(x)
        polys = [p0]
        
        if self.order > 0:
            p1 = 2 * x.clone()  # U_1(x) = 2x
            polys.append(p1)
            
            # Recursively compute higher-order polynomials
            for n in range(2, self.order + 1):
                pn = 2 * x * polys[n-1] - polys[n-2]
                polys.append(pn)
        
        return torch.stack(polys, dim=-1)

    def _gegenbauer_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Gegenbauer polynomial basis tensor.
        """
        constrained_alpha = self.activation(self.alpha).unsqueeze(0)  # Add batch dimension
        
        # Create first two polynomials: C_0(x) = 1
        p0 = torch.ones_like(x)
        polys = [p0]
        
        if self.order > 0:
            # Broadcast alpha for proper multiplication: [1, input_dim] * [batch_size, input_dim]
            p1 = 2 * constrained_alpha * x.clone()  # C_1(x) = 2αx
            polys.append(p1)
            
            # Recursively compute higher-order polynomials
            for n in range(2, self.order + 1):
                # Calculate coefficients with proper broadcasting
                coeff1 = 2 * (n + constrained_alpha - 1) / n
                coeff2 = (n + 2 * constrained_alpha - 2) / n
                
                pn = coeff1 * x * polys[n-1] - coeff2 * polys[n-2]
                polys.append(pn)
        
        return torch.stack(polys, dim=-1)

    def _hermite_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Physicists' Hermite polynomial basis tensor.
        """
        # Create first two polynomials: H_0(x) = 1
        p0 = torch.ones_like(x)
        polys = [p0]
        
        if self.order > 0:
            p1 = 2 * x.clone()  # H_1(x) = 2x
            polys.append(p1)
            
            # Recursively compute higher-order polynomials
            for n in range(2, self.order + 1):
                pn = 2 * x * polys[n-1] - 2 * (n - 1) * polys[n-2]
                polys.append(pn)
        
        return torch.stack(polys, dim=-1)

    def _laguerre_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Laguerre polynomial basis tensor.
        """
        # Create first two polynomials: L_0(x) = 1
        p0 = torch.ones_like(x)
        polys = [p0]
        
        if self.order > 0:
            p1 = 1 - x.clone()  # L_1(x) = 1 - x
            polys.append(p1)
            
            # Recursively compute higher-order polynomials
            for n in range(2, self.order + 1):
                coeff1 = (2 * n - 1) / n
                coeff2 = (n - 1) / n
                pn = (coeff1 - x) * polys[n-1] - coeff2 * polys[n-2]
                polys.append(pn)
        
        return torch.stack(polys, dim=-1)

    def _jacobi_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Jacobi polynomial basis tensor.
        """
        # Add batch dimension for broadcasting
        constrained_alpha = self.activation(self.alpha).unsqueeze(0)
        constrained_beta = self.activation(self.beta).unsqueeze(0)
        
        # Create first polynomial: P_0(x) = 1
        p0 = torch.ones_like(x)
        polys = [p0]
        
        if self.order > 0:
            # P_1(x) = 0.5 * ((α - β) + (α + β + 2)x)
            p1 = 0.5 * ((constrained_alpha - constrained_beta) + 
                       (constrained_alpha + constrained_beta + 2) * x.clone())
            polys.append(p1)
            
            # Recursively compute higher-order polynomials
            for n in range(2, self.order + 1):
                A = 2 * n * (n + constrained_alpha + constrained_beta) * (2 * n + constrained_alpha + constrained_beta - 2)
                B = (2 * n + constrained_alpha + constrained_beta - 1) * (
                    constrained_alpha**2 - constrained_beta**2 + 
                    x * (2 * n + constrained_alpha + constrained_beta - 2) * 
                    (2 * n + constrained_alpha + constrained_beta)
                )
                C = 2 * (n + constrained_alpha - 1) * (n + constrained_beta - 1) * (2 * n + constrained_alpha + constrained_beta)
                
                pn = (B * polys[n-1] - C * polys[n-2]) / A
                polys.append(pn)
        
        return torch.stack(polys, dim=-1)


class FourierBasis(BaseBasis):
    """Fourier basis with sine and cosine components."""

    def __init__(self, order: int):
        assert order >= 0, "Order must be a non-negative integer."
        super().__init__(order)

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the Fourier basis tensor for ``x``.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor scaled to ``[-1, 1]``.

        Returns
        -------
        torch.Tensor
            Tensor containing constant, cosine and sine components with size
            ``2*order + 1`` along the last dimension.
        """
        x_scaled = torch.clamp((x + 1), -1, 1) * torch.pi
        result = torch.zeros(*x.shape, 2 * self.order + 1, device=x.device)
        result[..., 0] = 1.0
        for k in range(1, self.order + 1):
            result[..., k] = torch.cos(k * x_scaled)
            result[..., k + self.order] = torch.sin(k * x_scaled)
        return result

    def calculate_fourier(self, x: torch.Tensor) -> torch.Tensor:
        warnings.warn(
            "calculate_fourier is deprecated; use calculate_basis instead",
            DeprecationWarning,
        )
        return self.calculate_basis(x)


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
        
    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the basis tensor using the selected activation function."""
        return self.ACTIVATION_WRAPPER[self.activation_type](x)

    def calculate_activation(self, x: torch.Tensor) -> torch.Tensor:
        warnings.warn(
            "calculate_activation is deprecated; use calculate_basis instead",
            DeprecationWarning,
        )
        return self.calculate_basis(x)
    
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
    """Prototype Haar wavelet basis for experimentation."""

    def __init__(self, order: int):
        super().__init__(order)

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_dim = x.shape
        wavelet_basis = torch.zeros(batch_size, input_dim, self.order, device=x.device)
        for i in range(self.order):
            wavelet_basis[..., i] = self.haar_wavelet(x, i)
        return wavelet_basis

    def haar_wavelet(self, x: torch.Tensor, order: int) -> torch.Tensor:
        if order == 0:
            return torch.ones_like(x)
        k = 2 ** (order - 1)
        return torch.where((x >= 0) & (x < 1 / k), 1, -1)
