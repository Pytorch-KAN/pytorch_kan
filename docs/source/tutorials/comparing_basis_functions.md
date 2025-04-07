# Comparing Basis Functions

This tutorial demonstrates how to compare different basis functions in Kolmogorov-Arnold Networks (KAN) to select the most appropriate one for a specific task.

## Introduction

KANs can use various orthogonal polynomials and other functions as basis functions. The choice of basis function can significantly affect the model's performance, convergence speed, and generalization capability. This tutorial will guide you through comparing different basis functions on a common task.

## Prerequisites

Before starting this tutorial, make sure you have:

1. Installed PyTorch KAN and its dependencies
2. Basic understanding of PyTorch and KANs
3. Familiarity with various basis functions

## Available Basis Functions

PyTorch KAN provides several built-in basis functions:

- `ChebyshevFirst`: First kind Chebyshev polynomials
- `ChebyshevSecond`: Second kind Chebyshev polynomials
- `Legendre`: Legendre polynomials
- `Hermite`: Hermite polynomials
- `Laguerre`: Laguerre polynomials
- `Gegenbauer`: Gegenbauer polynomials
- `Jacobi`: Jacobi polynomials

## Setting Up the Experiment

Let's set up an experiment to compare these basis functions on a function approximation task:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from time import time

from pytorch_kan.nn import KAN
from pytorch_kan.basis import (
    ChebyshevFirst, ChebyshevSecond, Legendre, 
    Hermite, Laguerre, Gegenbauer, Jacobi
)

# Define a target function to approximate
def target_function(x):
    return torch.sin(2 * np.pi * x) + 0.5 * torch.sin(4 * np.pi * x)

# Generate a dataset
x_train = torch.linspace(-1, 1, 500).unsqueeze(1)
y_train = target_function(x_train)

# Add some noise to make it more realistic
y_train = y_train + 0.05 * torch.randn_like(y_train)

# Create a test dataset (without noise)
x_test = torch.linspace(-1, 1, 100).unsqueeze(1)
y_test = target_function(x_test)

# Define the basis functions to compare
basis_functions = {
    'Chebyshev (First)': ChebyshevFirst(n_funcs=10),
    'Chebyshev (Second)': ChebyshevSecond(n_funcs=10),
    'Legendre': Legendre(n_funcs=10),
    'Hermite': Hermite(n_funcs=10),
    'Laguerre': Laguerre(n_funcs=10),
    'Gegenbauer': Gegenbauer(n_funcs=10, alpha=1.0),
    'Jacobi': Jacobi(n_funcs=10, alpha=1.0, beta=1.0)
}
```

## Training Function

Now, let's define a function to train a KAN model with a specific basis function:

```python
def train_and_evaluate(basis_name, basis_func, x_train, y_train, x_test, y_test, 
                       hidden_features=[32, 16], epochs=1000, lr=0.01):
    """
    Train a KAN model with the given basis function and evaluate its performance.
    
    Returns:
        tuple: (MSE on test set, training time, trained model)
    """
    # Create the model
    model = KAN(
        in_features=1,
        out_features=1,
        hidden_features=hidden_features,
        basis_func=basis_func
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    start_time = time()
    losses = []
    
    for epoch in range(epochs):
        # Forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            losses.append(loss.item())
            print(f'{basis_name} - Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
    
    training_time = time() - start_time
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(x_test)
        test_loss = criterion(test_outputs, y_test).item()
    
    print(f'{basis_name} - Test MSE: {test_loss:.6f}, Training Time: {training_time:.2f}s')
    return test_loss, training_time, model
```

## Running the Comparison

Let's run the experiment for each basis function:

```python
# Store results
results = {}
trained_models = {}

# Run the experiment for each basis function
for basis_name, basis_func in basis_functions.items():
    print(f"\nTraining with {basis_name} basis:")
    test_loss, training_time, model = train_and_evaluate(
        basis_name, basis_func, x_train, y_train, x_test, y_test
    )
    results[basis_name] = {
        'test_loss': test_loss,
        'training_time': training_time
    }
    trained_models[basis_name] = model
```

## Visualizing the Results

Now, let's visualize the performance of each basis function:

```python
# Plotting the comparison
def plot_performance_comparison(results):
    basis_names = list(results.keys())
    test_losses = [results[name]['test_loss'] for name in basis_names]
    training_times = [results[name]['training_time'] for name in basis_names]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot test losses
    bars1 = ax1.bar(basis_names, test_losses)
    ax1.set_title('Test MSE by Basis Function')
    ax1.set_xlabel('Basis Function')
    ax1.set_ylabel('Test MSE (lower is better)')
    ax1.set_ylim([0, max(test_losses) * 1.1])
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot training times
    bars2 = ax2.bar(basis_names, training_times)
    ax2.set_title('Training Time by Basis Function')
    ax2.set_xlabel('Basis Function')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_ylim([0, max(training_times) * 1.1])
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('basis_function_comparison.png')
    plt.show()

plot_performance_comparison(results)
```

## Visualizing the Learned Functions

Let's also visualize how well each model approximates the target function:

```python
def plot_function_approximations(trained_models, x_test, y_test):
    plt.figure(figsize=(12, 8))
    
    # Plot the ground truth
    plt.plot(x_test.numpy(), y_test.numpy(), 'k--', linewidth=2, label='Ground Truth')
    
    # Plot each model's prediction
    for basis_name, model in trained_models.items():
        with torch.no_grad():
            model.eval()
            y_pred = model(x_test)
        plt.plot(x_test.numpy(), y_pred.numpy(), label=basis_name)
    
    plt.title('Function Approximation Comparison')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('basis_function_approximations.png')
    plt.show()

plot_function_approximations(trained_models, x_test, y_test)
```

## Analyzing the Basis Functions

Different basis functions have different properties that make them suitable for different types of problems:

1. **Chebyshev polynomials** are good for approximating functions on bounded intervals like [-1, 1].
2. **Legendre polynomials** work well for uniform distributions.
3. **Hermite polynomials** are suited for problems with Gaussian distributions.
4. **Laguerre polynomials** are well-suited for problems on the positive real line.
5. **Gegenbauer and Jacobi polynomials** offer additional parameters to tune the basis functions.

## Choosing the Right Basis Function

The best basis function depends on your specific problem. Consider these factors:

1. **Domain of the problem**: Match the basis function's natural domain to your problem domain.
2. **Expected function shape**: Some basis functions better approximate certain function shapes.
3. **Convergence rate**: Different basis functions may converge faster for specific problems.
4. **Computational efficiency**: Some basis functions may be more computationally efficient.

## Conclusion

In this tutorial, we've seen how to compare different basis functions for KANs on a function approximation task. The choice of basis function can significantly impact model performance, and selecting the appropriate one for your problem is essential.

A systematic comparison like the one shown here can help you identify the most suitable basis function for your specific application, leading to better performance and efficiency in your KAN models.