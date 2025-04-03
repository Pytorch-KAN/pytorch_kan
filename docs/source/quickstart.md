# Quickstart Guide

This guide will help you get started with PyTorch KAN. We'll cover the basics of creating and training a KAN model with different basis functions.

## Basic Usage

Here's a simple example of how to use PyTorch KAN for a function approximation task:

```python
import torch
from pytorch_kan.nn import KAN
from pytorch_kan.basis import ChebyshevFirst

# Create a simple dataset
x = torch.linspace(-1, 1, 100).reshape(-1, 1)
y = torch.sin(x * 3.14159)

# Create a KAN model with Chebyshev polynomials of the first kind
model = KAN(
    input_dim=1,
    output_dim=1,
    basis_func=ChebyshevFirst(order=5),
    hidden_dim=32,
    num_layers=2
)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(x)
    test_loss = criterion(predictions, y)
    print(f'Test Loss: {test_loss.item():.4f}')
```

## Different Basis Functions

PyTorch KAN supports various basis functions. Here's how to use different ones:

```python
from pytorch_kan.basis import (
    ChebyshevFirst, ChebyshevSecond, Legendre, 
    Gegenbauer, Hermite, Laguerre, Jacobi
)

# Chebyshev polynomials of the first kind
model1 = KAN(input_dim=10, output_dim=1, basis_func=ChebyshevFirst(order=5))

# Chebyshev polynomials of the second kind
model2 = KAN(input_dim=10, output_dim=1, basis_func=ChebyshevSecond(order=5))

# Legendre polynomials
model3 = KAN(input_dim=10, output_dim=1, basis_func=Legendre(order=5))

# Gegenbauer polynomials (require alpha parameter)
model4 = KAN(input_dim=10, output_dim=1, 
            basis_func=Gegenbauer(order=5, alpha_size=10))

# Hermite polynomials
model5 = KAN(input_dim=10, output_dim=1, basis_func=Hermite(order=5))

# Laguerre polynomials
model6 = KAN(input_dim=10, output_dim=1, basis_func=Laguerre(order=5))

# Jacobi polynomials (require alpha and beta parameters)
model7 = KAN(input_dim=10, output_dim=1, 
            basis_func=Jacobi(order=5, alpha_size=10, beta_size=10))
```

## Input Scaling

For optimal performance, it's recommended to scale your input data to the range where the basis functions are most effective. PyTorch KAN provides scalers to help with this:

```python
from pytorch_kan.nn import MinMaxScaler, StandardScaler

# Min-Max Scaler (scales to [0,1] by default)
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Standard Scaler (zero mean, unit variance)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
```

## Matrix KAN Implementation

For higher performance on certain tasks, you can use the Matrix KAN implementation:

```python
from pytorch_kan.nn import MatrixKANLayer
import torch.nn as nn

class MatrixKANModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MatrixKANModel, self).__init__()
        self.layer = MatrixKANLayer(
            input_dim=input_dim, 
            output_dim=output_dim,
            spline_degree=3,    # Cubic splines
            grid_size=100       # Number of grid points
        )
    
    def forward(self, x):
        return self.layer(x)

model = MatrixKANModel(input_dim=10, output_dim=1)
```

For more detailed examples, check out the [tutorials section](tutorials/index).