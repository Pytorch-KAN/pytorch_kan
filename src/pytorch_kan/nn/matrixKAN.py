import torch
import torch.nn as nn
import torch.optim as optim
from src.pytorch_kan.basis.locals import *


class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, spline_degree=3, grid_size=100, grid_epsilon=1e-6):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.spline_degree = spline_degree
        self.grid_size = grid_size
        self.grid_epsilon = grid_epsilon

        # Precompute basis matrix for the given spline degree
        self.register_buffer('basis_matrix', binomial_coefficients_matrix(spline_degree))

        # Learnable polynomial coefficients with grid size dimension
        self.poly_matrix = nn.Parameter(torch.zeros(input_dim, output_dim, grid_size, spline_degree + 1))
        nn.init.kaiming_uniform_(self.poly_matrix, mode='fan_in', nonlinearity='relu')

        # Layer normalization for input standardization
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Normalize and scale input to [0,1] range
        x = self.norm(x)
        x = torch.clamp(x, -1.0 + self.grid_epsilon, 1.0 - self.grid_epsilon)
        x = (x + 1) / 2

        # Calculate grid indices and fractional positions
        indices = torch.floor(x * self.grid_size).long()
        t = (x * self.grid_size) - indices

        # Calculate power bases
        power_bases = torch.stack([t**i for i in range(self.spline_degree + 1)], dim=-1)
        
        # Compute basis values by multiplying with basis matrix
        basis_values = torch.matmul(power_bases, self.basis_matrix)
        
        # Prepare for efficient batch gathering of coefficients
        # Flatten the batch and input dimensions for indexing
        flat_input_indices = torch.arange(self.input_dim, device=x.device).repeat(batch_size)
        flat_indices = indices.reshape(-1)
        
        # Gather coefficients using flattened indices
        # Shape: [batch_size*input_dim, output_dim, spline_degree+1]
        flat_coeffs = self.poly_matrix[flat_input_indices, :, flat_indices, :]
        
        # Reshape back to original dimensions
        # Shape: [batch_size, input_dim, output_dim, spline_degree+1]
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