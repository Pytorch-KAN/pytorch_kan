"""Tests for data scaling utilities."""
import torch
import pytest
from pytorch_kan.nn import MinMaxScaler, StandardScaler, IdentityScaler

class TestScalers:
    """Test suite for data scaling utilities."""
    
    def test_minmax_scaler(self):
        """Test MinMaxScaler functionality."""
        # Create test data
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        
        # Initialize scaler
        scaler = MinMaxScaler()
        
        # Fit scaler
        scaler.fit(x)
        
        # Check computed statistics
        assert torch.allclose(scaler.min, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(scaler.max, torch.tensor([4.0, 5.0, 6.0]))
        
        # Transform data
        x_scaled = scaler.transform(x)
        
        # Expected result: scaled to [0, 1]
        expected = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        assert torch.allclose(x_scaled, expected)
        
        # Test inverse transform
        x_restored = scaler.inverse_transform(x_scaled)
        assert torch.allclose(x_restored, x)
        
        # Test fit_transform
        x_scaled_direct = scaler.fit_transform(x)
        assert torch.allclose(x_scaled_direct, expected)
    
    def test_standard_scaler(self):
        """Test StandardScaler functionality."""
        # Create test data
        x = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]], dtype=torch.float32)
        
        # Initialize scaler
        scaler = StandardScaler()
        
        # Fit scaler
        scaler.fit(x)
        
        # Check computed statistics
        assert torch.allclose(scaler.mean, torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(scaler.std, torch.tensor([1.0, 2.0, 3.0]))
        
        # Transform data
        x_scaled = scaler.transform(x)
        
        # Expected result: (x - mean) / std
        expected = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
        assert torch.allclose(x_scaled, expected)
        
        # Test inverse transform
        x_restored = scaler.inverse_transform(x_scaled)
        assert torch.allclose(x_restored, x)
        
        # Test fit_transform
        x_scaled_direct = scaler.fit_transform(x)
        assert torch.allclose(x_scaled_direct, expected)
    
    def test_identity_scaler(self):
        """Test IdentityScaler functionality."""
        # Create test data
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
        
        # Initialize scaler
        scaler = IdentityScaler()
        
        # Fit scaler (should be a no-op)
        scaler.fit(x)
        
        # Transform data (should return data unchanged)
        x_scaled = scaler.transform(x)
        assert torch.allclose(x_scaled, x)
        
        # Test inverse transform (should return data unchanged)
        x_restored = scaler.inverse_transform(x_scaled)
        assert torch.allclose(x_restored, x)
        
        # Test fit_transform (should return data unchanged)
        x_scaled_direct = scaler.fit_transform(x)
        assert torch.allclose(x_scaled_direct, x)