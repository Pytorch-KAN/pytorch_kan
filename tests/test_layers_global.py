import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from kan.nn.global_control.layers.fourier import FourierKANLayer
from kan.nn.global_control.layers.orthogonal import OrthogonalKANLayer
from kan.nn.global_control.layers.smooth_activation import SmoothActivationKANLayer
from kan.nn.global_control.layers.wavelet import WaveletKANLayer


def test_fourier_layer_forward_backward():
    layer = FourierKANLayer(input_dim=1, output_dim=2, order=2)
    x = torch.rand(5, 1, requires_grad=True) * 2 - 1
    out = layer(x)
    assert out.shape == (5, 2)
    out.sum().backward()
    assert layer.weights.grad is not None


def test_orthogonal_layer_forward_backward():
    layer = OrthogonalKANLayer(input_dim=1, output_dim=1, polynomial='legendre', order=2)
    x = torch.rand(3, 1, requires_grad=True) * 2 - 1
    out = layer(x)
    assert out.shape == (3, 1)
    out.sum().backward()
    assert layer.weights.grad is not None


def test_global_smooth_activation_layer_forward_backward():
    layer = SmoothActivationKANLayer(input_dim=1, output_dim=1, activation_type='silu')
    x = torch.randn(2, 1, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 1)
    out.sum().backward()
    assert layer.weights.grad is not None


def test_wavelet_layer_forward_backward():
    layer = WaveletKANLayer(input_dim=1, output_dim=1, order=2)
    x = torch.rand(4, 1, requires_grad=True)
    out = layer(x)
    assert out.shape == (4, 1)
    out.sum().backward()
    assert layer.weights.grad is not None
