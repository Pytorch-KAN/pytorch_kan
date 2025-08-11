import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pytest
from kan.nn.local_control.basis.bspline import BSplineBasis
from kan.nn.local_control.basis.rbf import RBFBasis
from kan.nn.local_control.basis.smooth_activation import SmoothActivationBasis


def test_bspline_basis_partition_of_unity():
    knots = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    x = torch.tensor([[1.5]])
    basis = BSplineBasis(order=4, degree=1, knots=knots)
    values = basis.calculate_basis(x)
    assert values.shape == (1, 1, 4)
    assert torch.allclose(values.sum(-1), torch.ones_like(x))


def test_rbf_basis_values():
    centers = torch.tensor([[0.0, 1.0]])  # (input_dim, num_centers)
    sigma = torch.tensor(1.0)
    x = torch.tensor([[0.0]])
    basis = RBFBasis(order=2, centers=centers)
    values = basis.calculate_basis(x, sigma=sigma)
    expected0 = torch.exp(-((x[0, 0] - centers[0, 0]) ** 2) / (sigma ** 2))
    expected1 = torch.exp(-((x[0, 0] - centers[0, 1]) ** 2) / (sigma ** 2))
    expected = torch.stack([expected0, expected1], dim=-1).unsqueeze(0).unsqueeze(0)
    assert values.shape == expected.shape
    assert torch.allclose(values, expected)


@pytest.mark.parametrize('activation', ['silu', 'gelu', 'softplus', 'mish', 'elu', 'tanh_shrink'])
def test_smooth_activation_basis_matches_pytorch(activation):
    x = torch.tensor([[0.5, -1.0]])
    basis = SmoothActivationBasis(order=1, activation_type=activation, alpha=1.0, beta=1.0)
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
