import torch
from functools import lru_cache

# For B-Splines

# The best option so far
@lru_cache(maxsize=None)
def binomial_coefficients_matrix(spline_degree):
    """
    Generate the matrix of binomial coefficients for a spline of a given degree.
    Each entry [i, j] corresponds to (-1)^(i-j) * C(i, j), where C(i, j) is the binomial coefficient.
    """
    # Initialize the basis matrix with zeros
    basis = torch.zeros(spline_degree + 1, spline_degree + 1)
    
    # Fill the matrix using the multiplicative formula for binomial coefficients
    for i in range(spline_degree + 1):
        for j in range(i + 1):  # Binomial coefficients are defined only for j <= i
            # Compute C(i, j) = i! / (j! * (i-j)!)
            coef = 1
            for k in range(1, j + 1):  # Multiplicative formula avoids factorial overflow
                coef *= (i - k + 1)
                coef //= k
            
            # Assign value with alternating sign
            basis[i, j] = coef * (-1)**(i - j)
    
    return basis

@lru_cache(maxsize=None)
def vandermonde_matrix(spline_degree):
    """
    Generate the Vandermonde matrix for a spline of a given degree, which is used to calculate the power bases of the spline.
    This is the method implemented in MatrixKAN for efficient KAN computation with splines.
    """

    t = torch.linspace(0, 1, spline_degree + 1)
    basis_matrix = torch.zeros((spline_degree + 1, spline_degree + 1))
    for i in range(spline_degree + 1):
        basis_matrix[i] = t ** i
    return torch.inverse(basis_matrix)


def cox_de_boor_basis(x, i, k, knots):
    """
    Evaluate the Cox-de Boor recursion formula for B-spline basis functions.
    
    Args:
        x: Input value
        i: Index of the basis function
        k: Degree of the spline
        knots: Knot vector
        
    Returns:
        Value of the B-spline basis function
    """
    if k == 0:
        return 1.0 if knots[i] <= x < knots[i+1] else 0.0
    
    # Handle division by zero cases
    d1 = knots[i+k] - knots[i]
    d2 = knots[i+k+1] - knots[i+1]
    
    f1 = 0.0 if d1 == 0.0 else (x - knots[i]) / d1
    f2 = 0.0 if d2 == 0.0 else (knots[i+k+1] - x) / d2
    
    return f1 * cox_de_boor_basis(x, i, k-1, knots) + f2 * cox_de_boor_basis(x, i+1, k-1, knots)


def de_boor_vectorized(x, knots, coefficients, degree):
    """
    Vectorized implementation of the de Boor algorithm for B-spline evaluation.
    
    Args:
        x: Tensor of input values [batch_size, input_dim]
        knots: Knot vector
        coefficients: B-spline coefficients
        degree: Degree of the spline
        
    Returns:
        Tensor of B-spline values
    """
    batch_size, input_dim = x.shape
    
    # Find knot span for each input value
    indices = torch.searchsorted(knots[degree:-degree], x.flatten()) - 1
    indices = torch.clamp(indices, 0, len(knots) - degree - 2)
    
    # Initialize coefficient array for the recursive algorithm
    d = torch.zeros((batch_size * input_dim, degree + 1), device=x.device)
    
    # For each point, get relevant coefficients
    for i in range(degree + 1):
        idx = indices + i
        d[:, i] = coefficients[idx]
    
    # Apply de Boor recursion
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            alpha = torch.zeros_like(x.flatten())
            
            # Calculate alpha (position between knots)
            left_knot = knots[indices + j - r]
            right_knot = knots[indices + j]
            denominator = right_knot - left_knot
            
            # Handle division by zero
            mask = denominator != 0
            alpha[mask] = (x.flatten()[mask] - left_knot[mask]) / denominator[mask]
            
            # Update coefficients
            d[:, j] = (1 - alpha) * d[:, j-1] + alpha * d[:, j]
    
    return d[:, degree].reshape(batch_size, input_dim)


def bspline_basis_expansion(x, degree, knots, num_bases):
    """
    Expand input tensor using B-spline basis functions. As used in BSRBF-KAN
    
    Args:
        x: Input tensor [batch_size, input_dim]
        degree: Degree of the spline
        knots: Knot vector
        num_bases: Number of basis functions
        
    Returns:
        Expanded tensor with B-spline basis values [batch_size, input_dim, num_bases]
    """
    batch_size, input_dim = x.shape
    basis_values = torch.zeros(batch_size, input_dim, num_bases, device=x.device)
    
    # For each basis function
    for i in range(num_bases):
        # Calculate the basis function values for all inputs
        for b in range(batch_size):
            for d in range(input_dim):
                basis_values[b, d, i] = cox_de_boor_basis(x[b, d], i, degree, knots)
    
    return basis_values


def rbf_basis(x, centers, sigma):

    batch_size, input_dim = x.shape
    num_rbf_bases = len(centers)

    rbf_values = torch.zeros(batch_size, input_dim, num_rbf_bases, device=x.device)
    for i, center in enumerate(centers):
        # Gaussian RBF: exp(-||x-c||²/σ²)
        squared_dist = torch.sum((x.unsqueeze(2) - center.view(1, 1, -1))**2, dim=-1)
        rbf_values[:, :, i] = torch.exp(-squared_dist / (sigma**2))

    return rbf_values


def bsrbf_expansion(x, degree, knots, centers, sigma):
    """
    Expand input tensor using both B-spline and Radial Basis Functions.
    
    Args:
        x: Input tensor [batch_size, input_dim]
        degree: Degree of the spline
        knots: Knot vector
        centers: Centers for RBF functions
        sigma: Width parameter for Gaussian RBFs
        
    Returns:
        Expanded tensor with combined basis values
    """
    num_bspline_bases = len(knots) - degree - 1
    
    # Calculate B-spline basis values
    bspline_values = bspline_basis_expansion(x, degree, knots, num_bspline_bases)
    
    # Calculate RBF basis values
    rbf_values = rbf_basis(x, centers, sigma)
    
    # Concatenate the basis values
    combined_values = torch.cat([bspline_values, rbf_values], dim=2)
    
    return combined_values


