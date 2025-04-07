# Changelog

All notable changes to the PyTorch KAN project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation using Sphinx
- Improved docstrings across all modules
- Tox configuration for multi-version Python testing
- Test suite for basis functions and neural network components

## [0.1.0] - 2025-04-01

### Added
- Initial release of PyTorch KAN
- Core implementation of Kolmogorov-Arnold Networks (KANs)
- Multiple basis function implementations:
  - Chebyshev polynomials (first and second kind)
  - Legendre polynomials
  - Gegenbauer polynomials
  - Hermite polynomials
  - Laguerre polynomials
  - Jacobi polynomials
  - B-splines
  - RBF (Radial Basis Functions)
- MatrixKANLayer implementation for efficient computation
- Data scaling utilities (MinMaxScaler, StandardScaler, IdentityScaler)
- MNIST classification examples for all basis function types
- Function approximation tutorials