import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from kan.nn.global_control.basis.orthogonal import OrthogonalPolynomial, FourierBasis, WaveletBasis, SmoothActivationBasis


def test_legendre_polynomial_values():
    x = torch.tensor([[0.5]])
    poly = OrthogonalPolynomial(polynomial='legendre', order=2)
    values = poly.calculate_basis(x)
    expected_p0 = torch.ones_like(x)
    expected_p1 = x
    expected_p2 = 0.5 * (3 * x ** 2 - 1)
    expected = torch.stack((expected_p0, expected_p1, expected_p2), dim=-1)
    assert values.shape == expected.shape
    assert torch.allclose(values, expected, atol=1e-5)


def test_fourier_basis_values():
    x = torch.tensor([[0.0]])
    basis = FourierBasis(order=2)
    values = basis.calculate_basis(x)
    expected = torch.tensor([[[1.0, -1.0, 1.0, 0.0, 0.0]]])
    assert values.shape == expected.shape
    assert torch.allclose(values, expected, atol=1e-6)


def test_wavelet_basis_shape_and_values():
    x = torch.tensor([[0.25]])
    basis = WaveletBasis(order=2)
    values = basis.calculate_basis(x)
    expected = torch.tensor([[[1.0, 1.0]]])
    assert values.shape == expected.shape
    assert torch.allclose(values, expected)


@pytest.mark.parametrize('activation', ['silu', 'gelu', 'softplus', 'mish', 'elu', 'tanh_shrink'])
def test_global_smooth_activation_basis(activation):
    x = torch.tensor([[0.5, -1.0]])
    basis = SmoothActivationBasis(activation_type=activation, alpha=1.0, beta=1.0)
    values = basis.calculate_basis(x)
    if activation == 'silu':
        expected = x * torch.sigmoid(x)
    elif activation == 'gelu':
        expected = torch.nn.functional.gelu(x)
    elif activation == 'softplus':
        expected = torch.nn.functional.softplus(x, beta=1.0)
    elif activation == 'mish':
        expected = x * torch.tanh(torch.nn.functional.softplus(x))
    elif activation == 'elu':
        expected = torch.nn.functional.elu(x, alpha=1.0)
    elif activation == 'tanh_shrink':
        expected = x - torch.tanh(x)
    assert torch.allclose(values, expected)
