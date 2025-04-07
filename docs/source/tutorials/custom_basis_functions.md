# Custom Basis Functions

This tutorial demonstrates how to create custom basis functions for Kolmogorov-Arnold Networks (KAN) in PyTorch.

## Introduction

While PyTorch KAN provides several built-in basis functions such as Chebyshev polynomials, Legendre polynomials, and more, you might need to create custom basis functions for specific applications. This tutorial will guide you through the process of implementing custom basis functions.

## Prerequisites

Before starting this tutorial, make sure you have:

1. Installed PyTorch KAN and its dependencies
2. Basic understanding of PyTorch and KANs
3. Familiarity with mathematical basis functions

## Creating a Custom Basis Class

To create a custom basis function for KAN, you need to extend the `BaseBasis` class and implement the required methods. Let's start by examining the structure of the `BaseBasis` class:

```python
from pytorch_kan.basis import BaseBasis
import torch

class CustomBasis(BaseBasis):
    """
    A custom basis function implementation.
    """
    
    def __init__(self, n_funcs=10, trainable=False):
        """
        Initialize the custom basis function.
        
        Args:
            n_funcs (int): Number of basis functions to use
            trainable (bool): Whether the basis parameters are trainable
        """
        super().__init__(n_funcs, trainable)
        # Add any additional parameters needed for your basis
        
    def forward(self, x):
        """
        Compute the basis function values for the given input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Basis function values of shape (batch_size, input_dim, n_funcs)
        """
        # Implement your custom basis function logic here
        batch_size, input_dim = x.shape
        out = torch.zeros(batch_size, input_dim, self.n_funcs, device=x.device)
        
        # Fill the output tensor with your basis function values
        # Example implementation:
        for i in range(self.n_funcs):
            out[:, :, i] = self._compute_basis_function(x, i)
            
        return out
    
    def _compute_basis_function(self, x, i):
        """
        Compute the i-th basis function value for the input x.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            i (int): Index of the basis function
            
        Returns:
            torch.Tensor: i-th basis function values of shape (batch_size, input_dim)
        """
        # Implement the specific computation for the i-th basis function
        # This is just an example, replace with your actual basis function
        return torch.sin((i + 1) * x * torch.pi)
```

## Example: Fourier Basis Functions

Let's implement a Fourier basis function as a practical example:

```python
class FourierBasis(BaseBasis):
    """
    Fourier basis functions using sines and cosines.
    """
    
    def __init__(self, n_funcs=10, trainable=False):
        """
        Initialize the Fourier basis.
        
        Args:
            n_funcs (int): Number of basis functions (should be even for equal sines and cosines)
            trainable (bool): Whether the basis parameters are trainable
        """
        super().__init__(n_funcs, trainable)
        # Make n_funcs even for equal number of sines and cosines
        if n_funcs % 2 != 0:
            self.n_funcs = n_funcs + 1
            print(f"Adjusted n_funcs to {self.n_funcs} to ensure equal number of sines and cosines")
        
        # Frequency parameter - can be made trainable
        self.freq = torch.nn.Parameter(
            torch.arange(1, self.n_funcs // 2 + 1, dtype=torch.float32), 
            requires_grad=trainable
        )
        
    def forward(self, x):
        """
        Compute the Fourier basis function values.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Basis function values of shape (batch_size, input_dim, n_funcs)
        """
        batch_size, input_dim = x.shape
        out = torch.zeros(batch_size, input_dim, self.n_funcs, device=x.device)
        
        # Scale x to be in [-π, π] range for better distribution of basis functions
        x_scaled = x * torch.pi
        
        # Compute sine basis functions (first half)
        for i in range(self.n_funcs // 2):
            out[:, :, i] = torch.sin(self.freq[i] * x_scaled)
        
        # Compute cosine basis functions (second half)
        for i in range(self.n_funcs // 2):
            out[:, :, i + self.n_funcs // 2] = torch.cos(self.freq[i] * x_scaled)
            
        return out
```

## Using Custom Basis Functions in KAN

Once you've defined your custom basis function, you can use it in a KAN model just like any built-in basis function:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from pytorch_kan.nn import KAN

# Create a dataset
x_train = torch.linspace(-1, 1, 500).unsqueeze(1)
y_train = torch.sin(2 * np.pi * x_train) + 0.5 * torch.sin(4 * np.pi * x_train)

# Create a KAN model with our custom Fourier basis
fourier_basis = FourierBasis(n_funcs=10, trainable=True)
model = KAN(
    in_features=1,
    out_features=1,
    hidden_features=[32, 16],
    basis_func=fourier_basis
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

# Visualize the results
model.eval()
with torch.no_grad():
    x_test = torch.linspace(-1, 1, 100).unsqueeze(1)
    y_pred = model(x_test)
    
plt.figure(figsize=(10, 6))
plt.plot(x_train.numpy(), y_train.numpy(), 'b.', label='Training Data')
plt.plot(x_test.numpy(), y_pred.numpy(), 'r-', linewidth=3, label='KAN Prediction')
plt.title('Function Approximation with Custom Fourier Basis')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('fourier_basis_approximation.png')
plt.show()
```

## Example: Wavelet Basis Functions

Another useful custom basis could be wavelets, which are particularly good at capturing localized features:

```python
class WaveletBasis(BaseBasis):
    """
    Wavelet basis functions using the Mexican hat wavelet.
    """
    
    def __init__(self, n_funcs=10, trainable=False):
        """
        Initialize the Wavelet basis.
        
        Args:
            n_funcs (int): Number of basis functions
            trainable (bool): Whether the basis parameters are trainable
        """
        super().__init__(n_funcs, trainable)
        
        # Centers of the wavelets, evenly distributed in [-1, 1]
        self.centers = torch.nn.Parameter(
            torch.linspace(-1, 1, self.n_funcs), 
            requires_grad=trainable
        )
        
        # Scales of the wavelets
        self.scales = torch.nn.Parameter(
            torch.ones(self.n_funcs) * 0.2,  # Default scale for good coverage
            requires_grad=trainable
        )
        
    def forward(self, x):
        """
        Compute the Wavelet basis function values.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            
        Returns:
            torch.Tensor: Basis function values of shape (batch_size, input_dim, n_funcs)
        """
        batch_size, input_dim = x.shape
        out = torch.zeros(batch_size, input_dim, self.n_funcs, device=x.device)
        
        for i in range(self.n_funcs):
            # Apply Mexican hat wavelet (second derivative of Gaussian)
            z = (x - self.centers[i]) / self.scales[i]
            z_squared = z ** 2
            out[:, :, i] = (1 - z_squared) * torch.exp(-z_squared / 2)
            
        return out
```

## Tips for Designing Custom Basis Functions

When designing custom basis functions, consider the following guidelines:

1. **Orthogonality**: When possible, use orthogonal basis functions to improve training efficiency.
2. **Smoothness**: Ensure basis functions have the appropriate level of smoothness for your application.
3. **Domain coverage**: Design basis functions that adequately cover the input domain.
4. **Trainable parameters**: Consider which parameters of your basis functions should be trainable.
5. **Computational efficiency**: Optimize for efficient forward and backward calculations.
6. **Numerical stability**: Implement your basis functions to be numerically stable.

## Testing Custom Basis Functions

It's essential to test your custom basis functions to ensure they behave as expected:

```python
def test_basis_function(basis_class, n_funcs=10, trainable=False):
    """
    Test a basis function implementation by visualizing its behavior.
    """
    # Initialize the basis
    basis = basis_class(n_funcs=n_funcs, trainable=trainable)
    
    # Generate a range of input values
    x = torch.linspace(-1, 1, 200).unsqueeze(1)
    
    # Compute the basis function values
    with torch.no_grad():
        y = basis(x)
    
    # Visualize each basis function
    plt.figure(figsize=(12, 6))
    for i in range(n_funcs):
        plt.plot(x.numpy(), y[:, 0, i].numpy(), label=f'Function {i+1}')
    
    plt.title(f'{basis_class.__name__} Basis Functions')
    plt.xlabel('x')
    plt.ylabel('Basis Function Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{basis_class.__name__}_visualization.png')
    plt.show()

# Test our custom basis functions
test_basis_function(FourierBasis, n_funcs=8)
test_basis_function(WaveletBasis, n_funcs=8)
```

## Conclusion

In this tutorial, we've learned how to create custom basis functions for Kolmogorov-Arnold Networks. By implementing your own basis functions, you can tailor KANs to specific applications and potentially improve performance on specialized tasks.

Custom basis functions allow you to incorporate domain knowledge into your KAN architecture, making them a powerful tool for specialized machine learning applications.