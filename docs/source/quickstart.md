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

## MNIST Classification Example

Here's a basic example of using a KAN for MNIST digit classification:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pytorch_kan.nn import KAN
from pytorch_kan.basis import ChebyshevFirst

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# Create a KAN model for MNIST classification
class KANClassifier(nn.Module):
    def __init__(self):
        super(KANClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.kan = KAN(
            input_dim=784,  # 28x28 MNIST images
            output_dim=10,  # 10 digit classes
            basis_func=ChebyshevFirst(order=3),
            hidden_dim=64,
            num_layers=2
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.kan(x)

# Initialize model, loss function, and optimizer
model = KANClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    # Test
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
```

For more detailed examples, check out the [tutorials section](tutorials/index).