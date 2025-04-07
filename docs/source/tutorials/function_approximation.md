# Function Approximation Tutorial

This tutorial demonstrates how to use PyTorch KAN for approximating functions. We'll focus on using different basis functions to approximate a sine wave.

## Setup

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_kan.nn import KAN
from pytorch_kan.basis import (
    ChebyshevFirst, ChebyshevSecond, Legendre, 
    Gegenbauer, Hermite, Laguerre
)
```

## Creating a Dataset

Let's create a simple sine wave dataset to approximate:

```python
# Create input data in range [-1, 1]
x = torch.linspace(-1, 1, 200).reshape(-1, 1)

# Generate target values (sine wave)
y = torch.sin(x * np.pi)

# Split into train and test sets
train_size = int(0.8 * len(x))
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(x_train.numpy(), y_train.numpy(), label='Training data')
plt.scatter(x_test.numpy(), y_test.numpy(), label='Test data')
plt.legend()
plt.title('Sine Function Dataset')
plt.xlabel('x')
plt.ylabel('sin(πx)')
plt.grid(True)
plt.show()
```

## Function to Train and Evaluate a KAN Model

```python
def train_and_evaluate_kan(basis_func, basis_name, epochs=2000):
    """Train and evaluate a KAN model with a specific basis function."""
    # Create a KAN model
    model = KAN(
        input_dim=1, 
        output_dim=1, 
        basis_func=basis_func,
        hidden_dim=16, 
        num_layers=2
    )
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        losses.append(loss.item())
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 500 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
        test_loss = criterion(y_pred, y_test)
        print(f'Test Loss for {basis_name}: {test_loss.item():.6f}')
    
    # Generate predictions for the entire range
    x_full = torch.linspace(-1, 1, 500).reshape(-1, 1)
    with torch.no_grad():
        y_pred_full = model(x_full)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot the loss curve
    plt.subplot(2, 1, 1)
    plt.plot(losses)
    plt.title(f'Training Loss - {basis_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.yscale('log')
    
    # Plot the predictions vs actual
    plt.subplot(2, 1, 2)
    plt.scatter(x_train.numpy(), y_train.numpy(), s=10, alpha=0.6, label='Training data')
    plt.scatter(x_test.numpy(), y_test.numpy(), s=10, alpha=0.6, label='Test data')
    plt.plot(x_full.numpy(), y_pred_full.numpy(), 'r-', linewidth=2, label='KAN prediction')
    plt.plot(x_full.numpy(), torch.sin(x_full * np.pi).numpy(), 'g--', linewidth=2, label='True function')
    plt.title(f'Function Approximation with {basis_name}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, test_loss.item()
```

## Comparing Different Basis Functions

Let's compare how different basis functions perform for this task:

```python
# Dictionary to store test losses for comparison
results = {}

# Try different basis functions
basis_functions = [
    (ChebyshevFirst(order=5), "Chebyshev First Kind"),
    (ChebyshevSecond(order=5), "Chebyshev Second Kind"),
    (Legendre(order=5), "Legendre"),
    (Gegenbauer(order=5, alpha_size=1), "Gegenbauer"),
    (Hermite(order=5), "Hermite"),
    (Laguerre(order=5), "Laguerre")
]

for basis_func, basis_name in basis_functions:
    print(f"\nTraining with {basis_name} basis:")
    _, loss = train_and_evaluate_kan(basis_func, basis_name)
    results[basis_name] = loss

# Compare results
plt.figure(figsize=(10, 6))
names = list(results.keys())
values = list(results.values())
plt.bar(names, values)
plt.title('Test Loss Comparison Between Basis Functions')
plt.xlabel('Basis Function')
plt.ylabel('Test Loss (MSE)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

## Analysis

In this tutorial, we've seen how different basis functions perform when approximating a sine wave:

1. **Chebyshev polynomials (first kind)** are particularly well-suited for approximating functions on bounded intervals like [-1, 1], which matches our sine wave domain.

2. **Legendre polynomials** also perform well on bounded intervals and can provide good approximation for smooth functions.

3. **Hermite polynomials** are typically better for functions defined on unbounded domains, so they might not be optimal for this particular task.

4. **Laguerre polynomials** are designed for functions on [0, ∞), so they may require input transformations for best results on our [-1, 1] domain.

The choice of basis function should depend on the specific properties of the function you're trying to approximate. Experiment with different options and parameters to find the best fit for your application.

## Increasing Approximation Accuracy

To improve the approximation accuracy, you can try:

1. Increasing the order of the basis functions
2. Adding more layers to the KAN model
3. Increasing the hidden dimension
4. Training for more epochs
5. Adjusting the learning rate

```python
# Example of a more complex model
advanced_model = KAN(
    input_dim=1, 
    output_dim=1, 
    basis_func=ChebyshevFirst(order=10),  # Higher order
    hidden_dim=32,                        # Larger hidden dimension
    num_layers=3                          # More layers
)
```