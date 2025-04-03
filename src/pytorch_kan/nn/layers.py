import torch
import torch.nn as nn
import torch.optim as optim
import math
from src.basis.locals import BSplineBasis


"""
Kolmogorov-Arnold Network Layers

This module implements different neural network layers according to various approaches 
of the Kolmogorov-Arnold representation theorem. These layers form the building blocks
for constructing complete Kolmogorov-Arnold Networks (KANs).
"""

class MatrixKANLayer(nn.Module):
    """
    Matrix-based implementation of a Kolmogorov-Arnold Network layer using B-splines.
    
    This layer approximates functions using piecewise polynomial representations 
    through a grid-based approach. It uses B-splines for smooth interpolation between
    grid points and efficiently implements the Kolmogorov-Arnold representation.
    
    The layer transforms each input dimension separately using B-spline basis functions,
    then combines them using learnable coefficients to produce the output.
    
    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        spline_degree (int): Degree of the B-spline polynomials (e.g., 3 for cubic).
        grid_size (int): Number of grid points for discretizing the input space.
        grid_epsilon (float): Small value to prevent boundary issues in clamping.
        basis_matrix (torch.Tensor): Precomputed basis matrix for the B-splines.
        poly_matrix (nn.Parameter): Learnable polynomial coefficients.
        norm (nn.LayerNorm): Layer normalization for input standardization.
    """
    def __init__(self, input_dim, output_dim, spline_degree=3, grid_size=100, grid_epsilon=1e-6):
        """
        Initialize a MatrixKANLayer.
        
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            spline_degree (int, optional): Degree of the B-spline polynomials. Defaults to 3 (cubic).
            grid_size (int, optional): Number of grid points for discretizing the input space. 
                                      Defaults to 100.
            grid_epsilon (float, optional): Small value to prevent boundary issues in clamping. 
                                           Defaults to 1e-6.
        """
        super(MatrixKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spline_degree = spline_degree
        self.grid_size = grid_size
        self.grid_epsilon = grid_epsilon

        # Precompute basis matrix for the given spline degree
        self.register_buffer('basis_matrix', BSplineBasis._binomial_coefficients_matrix(spline_degree))

        # Learnable polynomial coefficients with grid size dimension
        self.poly_matrix = nn.Parameter(torch.zeros(input_dim, output_dim, grid_size, spline_degree + 1))
        nn.init.kaiming_uniform_(self.poly_matrix, mode='fan_in', nonlinearity='relu')

        # Layer normalization for input standardization
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        """
        Forward pass of the MatrixKANLayer.
        
        This method:
        1. Normalizes the input
        2. Maps input to grid indices and fractional positions
        3. Calculates B-spline basis values
        4. Computes weighted combinations using learnable coefficients
        5. Sums across input dimensions to produce the final output
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        batch_size = x.size(0)
        
        # Normalize and scale input to [0,1] range
        x = self.norm(x)
        x = torch.clamp(x, -1.0 + self.grid_epsilon, 1.0 - self.grid_epsilon)
        x = (x + 1) / 2

        # Calculate grid indices and fractional positions
        indices = torch.floor(x * self.grid_size).long()
        t = (x * self.grid_size) - indices

        # Calculate power bases [1, t, t^2, ..., t^(spline_degree)]
        power_bases = torch.stack([t**i for i in range(self.spline_degree + 1)], dim=-1)
        
        # Compute basis values by multiplying with basis matrix
        basis_values = torch.matmul(power_bases, self.basis_matrix)
        
        # Prepare for efficient batch gathering of coefficients
        # Flatten the batch and input dimensions for indexing if input_dim = 3 [0, 1, 2] * batch_size
        flat_input_indices = torch.arange(self.input_dim, device=x.device).repeat(batch_size)
        flat_indices = indices.reshape(-1)
        
        # Gather coefficients using flattened indices
        # Shape: [batch_size*input_dim, output_dim, spline_degree+1] Select the specific input_dimension, all the output_dimension, the sepcific grid_Dimension and all the spline_degree dimension
        flat_coeffs = self.poly_matrix[flat_input_indices, :, flat_indices, :]
        
        # Reshape back to original dimensions
        # Shape: [batch_size, self.input_dim, self.output_dim, self.spline_degree + 1]
        gathered_coeffs = flat_coeffs.reshape(batch_size, self.input_dim, self.output_dim, self.spline_degree + 1)
        
        # Expand basis values for broadcasting
        # Shape: [batch_size, input_dim, 1, spline_degree+1]
        basis_values_expanded = basis_values.unsqueeze(2)
        
        # Compute weighted basis values
        # Shape: [batch_size, input_dim, output_dim]
        weighted_basis = (basis_values_expanded * gathered_coeffs).sum(dim=-1)
        
        # Sum across input dimension to get final output
        # Shape: [batch_size, output_dim]
        output = weighted_basis.sum(dim=1)
        
        return output


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer using general basis functions.
    
    This layer implements a single layer of the Kolmogorov-Arnold Network using
    any basis function that follows the BaseBasis interface. It transforms each input
    dimension using the specified basis functions and combines them with learnable
    weights to produce the output.
    
    This implementation follows the general formulation of the Kolmogorov-Arnold 
    representation theorem, which states that any multivariate continuous function 
    can be represented as a superposition of continuous functions of a single variable.
    
    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        basis_func (BaseBasis): Basis function to transform inputs.
        bias (bool): Whether to include a bias term.
        weights (nn.Parameter): Learnable weights for combining basis functions.
        bias_param (nn.Parameter, optional): Learnable bias parameters.
    """
    def __init__(self, in_features, out_features, basis_func, bias=True):
        """
        Initialize a KANLayer.
        
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            basis_func (BaseBasis): Basis function object to transform inputs.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
        """
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_func = basis_func
        self.basis_dim = basis_func.order + 1  # Number of basis functions
        
        # Initialize weights with shape [out_features, in_features, basis_dim]
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features, self.basis_dim))
        
        if bias:
            self.bias_param = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_param', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights and bias using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias_param is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_param, -bound, bound)
    
    def forward(self, x):
        """
        Forward pass of the KANLayer.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_features].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_features].
        """
        # Calculate basis functions for each input feature
        # Output shape: [batch_size, in_features, basis_dim]
        basis_values = self.basis_func.calculate_basis(x)
        
        # Apply learnable weights to basis values
        # Multiply: [batch_size, in_features, basis_dim] x [out_features, in_features, basis_dim]
        # Output: [batch_size, out_features, in_features, basis_dim]
        weighted_basis = basis_values.unsqueeze(1) * self.weights.unsqueeze(0)
        
        # Sum over basis dimension and input features
        # Output: [batch_size, out_features]
        output = weighted_basis.sum(dim=(2, 3))
        
        if self.bias_param is not None:
            output = output + self.bias_param
            
        return output
    
    def extra_repr(self):
        """Return a string representation of the layer parameters."""
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias_param is not None
        )


class KAN(nn.Module):
    """
    Complete Kolmogorov-Arnold Network implementation.
    
    A KAN is a neural network architecture based on the Kolmogorov-Arnold representation
    theorem, which can approximate any continuous multivariate function using compositions
    of continuous functions of a single variable.
    
    This implementation allows for constructing multi-layer KANs with various basis functions,
    supporting a wide range of function approximation tasks.
    
    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        basis_func (BaseBasis): Basis function to transform inputs.
        hidden_dim (int): Number of hidden units in each layer.
        num_layers (int): Number of KAN layers in the network.
        use_bias (bool): Whether to include bias terms in the layers.
        layers (nn.ModuleList): List of KANLayer modules.
    """
    def __init__(self, input_dim, output_dim, basis_func, hidden_dim=64, num_layers=1, use_bias=True):
        """
        Initialize a Kolmogorov-Arnold Network.
        
        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            basis_func (BaseBasis): Basis function object to transform inputs.
            hidden_dim (int, optional): Number of hidden units in each layer. Defaults to 64.
            num_layers (int, optional): Number of KAN layers. Defaults to 1.
            use_bias (bool, optional): Whether to include bias terms. Defaults to True.
        """
        super(KAN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.basis_func = basis_func
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_bias = use_bias
        
        # Build the layers
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(KANLayer(input_dim, hidden_dim if num_layers > 1 else output_dim, 
                                   basis_func, use_bias))
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.layers.append(KANLayer(hidden_dim, hidden_dim, basis_func, use_bias))
            
        # Output layer (if there are hidden layers)
        if num_layers > 1:
            self.layers.append(KANLayer(hidden_dim, output_dim, basis_func, use_bias))
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim].
        """
        for layer in self.layers:
            x = layer(x)
        return x
    
    def extra_repr(self):
        """Return a string representation of the network parameters."""
        return 'input_dim={}, hidden_dim={}, output_dim={}, num_layers={}'.format(
            self.input_dim, self.hidden_dim, self.output_dim, self.num_layers
        )

