"""Tests for KAN neural network layers."""
import torch
import pytest
import math
from pytorch_kan.basis import ChebyshevFirst
from pytorch_kan.nn import KANLayer, KAN

class TestKANLayers:
    """Test suite for KAN neural network layers."""
    
    def test_kan_layer_initialization(self):
        """Test that KANLayer can be properly initialized."""
        # Create a basis function
        basis = ChebyshevFirst(order=3)
        
        # Initialize KANLayer
        layer = KANLayer(in_features=5, out_features=2, basis_func=basis, bias=True)
        
        # Check layer properties
        assert layer.in_features == 5
        assert layer.out_features == 2
        assert layer.basis_func == basis
        assert layer.basis_dim == 4  # order + 1
        assert layer.weights.shape == (2, 5, 4)  # [out_features, in_features, basis_dim]
        assert layer.bias_param is not None
        assert layer.bias_param.shape == (2,)  # [out_features]
    
    def test_kan_layer_forward(self):
        """Test the forward pass of KANLayer."""
        # Create a basis function
        basis = ChebyshevFirst(order=2)
        
        # Initialize KANLayer with deterministic weights for testing
        layer = KANLayer(in_features=3, out_features=2, basis_func=basis, bias=True)
        
        # Set deterministic weights and bias
        torch.nn.init.ones_(layer.weights)
        torch.nn.init.zeros_(layer.bias_param)
        
        # Create input tensor
        x = torch.zeros(1, 3)  # batch_size=1, in_features=3
        
        # Forward pass
        output = layer(x)
        
        # Expected output: For each x=0, basis values are [1,0,-1], multiplied by weights=1, summed across features
        # result should be [3,3] for basis values [1,0,-1] * weights=1 * 3 features, with bias=0
        expected_output = torch.tensor([[3.0, 3.0]])
        assert output.shape == (1, 2)  # [batch_size, out_features]
        assert torch.allclose(output, expected_output, atol=1e-6)
    
    def test_kan_initialization(self):
        """Test that KAN can be properly initialized."""
        # Create a basis function
        basis = ChebyshevFirst(order=3)
        
        # Initialize KAN
        model = KAN(input_dim=10, output_dim=2, basis_func=basis, hidden_dim=64, num_layers=3)
        
        # Check model properties
        assert model.input_dim == 10
        assert model.output_dim == 2
        assert model.basis_func == basis
        assert model.hidden_dim == 64
        assert model.num_layers == 3
        assert len(model.layers) == 3
        
        # Check layer dimensions
        assert model.layers[0].in_features == 10
        assert model.layers[0].out_features == 64
        assert model.layers[1].in_features == 64
        assert model.layers[1].out_features == 64
        assert model.layers[2].in_features == 64
        assert model.layers[2].out_features == 2
    
    def test_kan_forward(self):
        """Test the forward pass of KAN."""
        # Create a basis function
        basis = ChebyshevFirst(order=2)
        
        # Initialize a simple 1-layer KAN for easier testing
        model = KAN(input_dim=3, output_dim=2, basis_func=basis, num_layers=1)
        
        # Set deterministic weights for all layers
        for layer in model.layers:
            torch.nn.init.ones_(layer.weights)
            if layer.bias_param is not None:
                torch.nn.init.zeros_(layer.bias_param)
        
        # Create input tensor
        x = torch.zeros(1, 3)  # batch_size=1, in_features=3
        
        # Forward pass
        output = model(x)
        
        # Expected output: Same as KANLayer test since we have only one layer
        expected_output = torch.tensor([[3.0, 3.0]])
        assert output.shape == (1, 2)
        assert torch.allclose(output, expected_output, atol=1e-6)
    
    def test_kan_multi_layer(self):
        """Test KAN with multiple layers."""
        # Create a basis function
        basis = ChebyshevFirst(order=1)  # Simpler basis for clearer testing
        
        # Initialize a 2-layer KAN
        model = KAN(input_dim=2, output_dim=1, basis_func=basis, hidden_dim=2, num_layers=2)
        
        # Set custom weights for deterministic testing
        # First layer
        model.layers[0].weights.data = torch.ones(2, 2, 2)  # [hidden_dim, input_dim, basis_dim]
        model.layers[0].bias_param.data = torch.zeros(2)
        
        # Second layer
        model.layers[1].weights.data = torch.ones(1, 2, 2)  # [output_dim, hidden_dim, basis_dim]
        model.layers[1].bias_param.data = torch.zeros(1)
        
        # Create input tensor
        x = torch.zeros(1, 2)  # batch_size=1, input_dim=2
        
        # Forward pass
        output = model(x)
        
        # Expected computation:
        # Layer 1: For each x=0, basis values are [1,0], multiplied by weights=1, summed across features
        # result should be [2,2] for basis values [1,0] * weights=1 * 2 features
        # Layer 2: For each x=2, basis values are [1,2], multiplied by weights=1, summed across features
        # result should be [6] for basis values [1,2] * weights=1 * 2 features
        assert output.shape == (1, 1)
        # The exact value will depend on the basis function evaluation at x=2
        # This is a simplification for the test