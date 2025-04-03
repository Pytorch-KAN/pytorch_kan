"""Tests for orthogonal polynomial basis functions."""
import torch
import pytest
from pytorch_kan.basis import (
    ChebyshevFirst,
    ChebyshevSecond,
    Legendre,
    Gegenbauer,
    Hermite,
    Laguerre,
    Jacobi
)

class TestOrthogonalPolynomials:
    """Test suite for orthogonal polynomial basis functions."""
    
    def test_chebyshev_first(self):
        """Test Chebyshev polynomial of the first kind."""
        basis = ChebyshevFirst(order=3)
        x = torch.tensor([[-0.5, 0.0, 0.5]], dtype=torch.float32)
        result = basis.calculate_basis(x)
        
        # Shape should be [batch_size, input_dim, order+1]
        assert result.shape == (1, 3, 4)
        
        # Test specific values (T_0(x) = 1, T_1(x) = x, T_2(x) = 2x² - 1, T_3(x) = 4x³ - 3x)
        # For x = 0.5:
        expected_values = torch.tensor([
            [1.0, 0.5, -0.5, -0.875]  # For x = 0.5
        ], dtype=torch.float32)
        assert torch.allclose(result[0, 2], expected_values, atol=1e-6)
    
    def test_legendre(self):
        """Test Legendre polynomials."""
        basis = Legendre(order=2)
        x = torch.tensor([[0.0, 0.5]], dtype=torch.float32)
        result = basis.calculate_basis(x)
        
        # Shape should be [batch_size, input_dim, order+1]
        assert result.shape == (1, 2, 3)
        
        # Test specific values (P_0(x) = 1, P_1(x) = x, P_2(x) = (3x² - 1)/2)
        # For x = 0.5:
        expected_values = torch.tensor([
            [1.0, 0.5, 0.125]  # For x = 0.5
        ], dtype=torch.float32)
        assert torch.allclose(result[0, 1], expected_values, atol=1e-6)
    
    @pytest.mark.parametrize("basis_class,params", [
        (ChebyshevFirst, {"order": 5}),
        (ChebyshevSecond, {"order": 4}),
        (Legendre, {"order": 3}),
        (Gegenbauer, {"order": 3, "alpha_size": 1}),
        (Hermite, {"order": 4}),
        (Laguerre, {"order": 3}),
        (Jacobi, {"order": 2, "alpha_size": 1, "beta_size": 1}),
    ])
    def test_initialization(self, basis_class, params):
        """Test that basis functions can be initialized."""
        basis = basis_class(**params)
        assert basis.order == params["order"]
        
    @pytest.mark.parametrize("basis_class,params", [
        (ChebyshevFirst, {"order": 5}),
        (ChebyshevSecond, {"order": 4}),
        (Legendre, {"order": 3}),
        (Gegenbauer, {"order": 3, "alpha_size": 1}),
        (Hermite, {"order": 4}),
        (Laguerre, {"order": 3}),
        (Jacobi, {"order": 2, "alpha_size": 1, "beta_size": 1}),
    ])
    def test_output_shape(self, basis_class, params):
        """Test that basis functions produce the correct output shape."""
        basis = basis_class(**params)
        x = torch.randn(2, 3)  # batch_size=2, input_dim=3
        result = basis.calculate_basis(x)
        
        expected_shape = (2, 3, params["order"] + 1)
        assert result.shape == expected_shape