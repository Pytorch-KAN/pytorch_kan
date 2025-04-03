"""Tests for the base basis functions."""
import torch
import pytest
from pytorch_kan.basis import BaseBasis

def test_abstract_basis():
    """Test that BaseBasis is properly defined as an abstract class."""
    with pytest.raises(TypeError):
        # Should raise TypeError because BaseBasis is abstract
        BaseBasis(order=5)

def test_basis_initialization():
    """Test that concrete basis classes can be properly initialized."""
    # This will be expanded when we implement concrete test cases
    pass