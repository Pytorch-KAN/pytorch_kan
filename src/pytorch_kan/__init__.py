"""
PyTorch KAN: Kolmogorov-Arnold Networks

This package provides an efficient and scalable implementation of Kolmogorov-Arnold Networks (KANs),
a neural network architecture based on the Kolmogorov-Arnold representation theorem.

The package contains modules for building KANs with various basis functions, including:
- Orthogonal polynomials (Chebyshev, Legendre, etc.)
- B-splines
- Radial basis functions
- Fourier basis
- Wavelet basis
- Smooth activation functions

Main components:
- basis: Basis function implementations
- nn: Neural network modules for building KANs
"""

from pytorch_kan.basis import *
from pytorch_kan.nn import *

__version__ = "1.0.0"
__author__ = "PyTorch KAN Team"