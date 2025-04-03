# nn

Base modules for building Kolmogorov-Arnold Networks

# Neural Network Components

## Overview

This module provides the foundational neural network components for building Kolmogorov-Arnold Networks (KANs). These components are designed to work with various basis functions to create flexible and powerful KAN architectures.

## Key Components

### KAN

The core implementation of Kolmogorov-Arnold Networks that combines multiple basis functions to approximate complex functions.

```python
from pytorch_kan.nn import KAN
from pytorch_kan.basis import ChebyshevFirst

model = KAN(
    input_dim=10,
    output_dim=1,
    basis_func=ChebyshevFirst(),
    hidden_dim=64
)
```

### KANLayer

A single layer implementation of the Kolmogorov-Arnold representation that can be stacked or combined with other PyTorch layers.

```python
from pytorch_kan.nn import KANLayer

layer = KANLayer(
    in_features=10,
    out_features=5,
    basis_func=basis_function,
    bias=True
)
```

## Customization

The neural network components are designed to be highly customizable:

- Use with any basis function from the `pytorch_kan.basis` module
- Compatible with PyTorch's standard optimization methods
- Easily integrated into larger network architectures

## Performance Considerations

- KAN layers typically require more parameters than traditional neural networks for the same number of neurons
- Consider using `torch.jit.script` to optimize inference performance
- Batch normalization can help stabilize training with certain basis functions