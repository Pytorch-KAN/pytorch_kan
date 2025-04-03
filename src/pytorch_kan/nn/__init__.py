"""
Neural Network Components for Kolmogorov-Arnold Networks

This module provides the foundational neural network components for building
Kolmogorov-Arnold Networks (KANs), which implement the Kolmogorov-Arnold
representation theorem as a neural network architecture.

Key components:
- KAN: Complete Kolmogorov-Arnold Network implementation
- KANLayer: Single layer for Kolmogorov-Arnold representation
- Scaler: Input normalization and scaling utilities

These components can be used with various basis functions from the pytorch_kan.basis
module to create flexible and powerful KAN architectures for function approximation.
"""

from pytorch_kan.nn.layers import KAN, KANLayer
from pytorch_kan.nn.scaler import MinMaxScaler, StandardScaler, IdentityScaler

__all__ = [
    'KAN',
    'KANLayer',
    'MinMaxScaler',
    'StandardScaler',
    'IdentityScaler',
]