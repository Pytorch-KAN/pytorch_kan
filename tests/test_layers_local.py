import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from kan.nn.local_control.layers.rbf import RBFKANLayer
from kan.nn.local_control.layers.matrix import MatrixKANLayer
from kan.nn.local_control.layers.smooth_activation import SmoothActivationKANLayer


def test_rbf_layer_forward_backward():
    layer = RBFKANLayer(input_dim=1, output_dim=1, num_centers=2, sigma=1.0)
    x = torch.randn(4, 1, requires_grad=True)
    out = layer(x)
    assert out.shape == (4, 1)
    out.sum().backward()
    assert layer.centers.grad is not None
    assert layer.weights.grad is not None


def test_matrix_layer_forward_backward():
    layer = MatrixKANLayer(input_dim=1, output_dim=1, spline_degree=1, grid_size=4)
    x = torch.rand(2, 1, requires_grad=True) * 2 - 1
    out = layer(x)
    assert out.shape == (2, 1)
    out.sum().backward()
    assert layer.poly_matrix.grad is not None


def test_smooth_activation_layer_forward_backward():
    layer = SmoothActivationKANLayer(input_dim=2, output_dim=1, activation_type='silu')
    x = torch.randn(3, 2, requires_grad=True)
    out = layer(x)
    assert out.shape == (3, 1)
    out.sum().backward()
    assert layer.weights.grad is not None
