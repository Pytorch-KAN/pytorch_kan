"""
Basis Functions for Kolmogorov-Arnold Networks

This module provides various basis function implementations for Kolmogorov-Arnold Networks (KANs).
It includes both global control basis functions (globals.py) and local control basis functions (locals.py).

Global control basis functions apply transformations across the entire input domain, providing
broader approximation capability for capturing overall patterns in the data.

Local control basis functions focus on specific regions of the input domain, allowing for more
detailed approximation in those areas, particularly useful for capturing fine details or handling discontinuities.

Available basis functions:
- Orthogonal polynomials: Legendre, Chebyshev, Gegenbauer, Hermite, Laguerre, Jacobi
- Fourier basis: Sine and cosine functions
- Wavelet basis: Haar wavelets
- Smooth activation functions: SiLU, GELU, Softplus, Mish, ELU, TanhShrink
- B-splines: Local piecewise polynomial basis functions
- Radial basis functions: Gaussian RBFs centered at different points
"""

from pytorch_kan.basis.base import BaseBasis
from pytorch_kan.basis.globals import (
    OrthogonalPolynomial, 
    FourierBasis, 
    SmoothActivationBasis as GlobalSmoothActivationBasis,
    WaveletBasis
)
from pytorch_kan.basis.locals import (
    BSplineBasis, 
    RBFBasis, 
    SmoothActivationBasis as LocalSmoothActivationBasis
)

# Convenient aliases for specific orthogonal polynomials
def ChebyshevFirst(order, **kwargs):
    """Chebyshev polynomial of the first kind (T_n) basis."""
    return OrthogonalPolynomial("chebyshev_first", order, **kwargs)

def ChebyshevSecond(order, **kwargs):
    """Chebyshev polynomial of the second kind (U_n) basis."""
    return OrthogonalPolynomial("chebyshev_second", order, **kwargs)

def Legendre(order, **kwargs):
    """Legendre polynomial (P_n) basis."""
    return OrthogonalPolynomial("legendre", order, **kwargs)

def Gegenbauer(order, alpha_size, **kwargs):
    """Gegenbauer (ultraspherical) polynomial basis with parameter alpha."""
    return OrthogonalPolynomial("gegenbauer", order, alpha_size=alpha_size, **kwargs)

def Hermite(order, **kwargs):
    """Physicists' Hermite polynomial (H_n) basis."""
    return OrthogonalPolynomial("hermite", order, **kwargs)

def Laguerre(order, **kwargs):
    """Laguerre polynomial (L_n) basis for functions on [0, ∞)."""
    return OrthogonalPolynomial("laguerre", order, **kwargs)

def Jacobi(order, alpha_size, beta_size, **kwargs):
    """Jacobi polynomial basis with parameters alpha and beta."""
    return OrthogonalPolynomial("jacobi", order, alpha_size=alpha_size, beta_size=beta_size, **kwargs)

__all__ = [
    'BaseBasis',
    'OrthogonalPolynomial',
    'FourierBasis',
    'WaveletBasis',
    'BSplineBasis',
    'RBFBasis',
    'ChebyshevFirst',
    'ChebyshevSecond',
    'Legendre',
    'Gegenbauer',
    'Hermite',
    'Laguerre',
    'Jacobi',
]