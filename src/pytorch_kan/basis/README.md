# Basis

Methods and Classes for creating the basis of Kolmogorov-Arnold Networks from the input tensors

# Basis Functions

## Overview

This module provides a collection of basis functions for Kolmogorov-Arnold Networks (KANs). These functions form the mathematical foundation of KANs, allowing them to approximate a wide range of complex functions according to the Kolmogorov-Arnold representation theorem.

## Available Basis Functions

### Chebyshev Polynomials

#### First Kind (ChebyshevFirst)
- Defined by the recurrence relation: T₀(x) = 1, T₁(x) = x, Tₙ₊₁(x) = 2xTₙ(x) - Tₙ₋₁(x)
- Orthogonal with respect to the weight function (1-x²)^(-1/2)
- Excellent for approximating functions on bounded intervals

```python
from pytorch_kan.basis import ChebyshevFirst

basis = ChebyshevFirst(degree=5)
```

#### Second Kind (ChebyshevSecond)
- Defined by the recurrence relation: U₀(x) = 1, U₁(x) = 2x, Uₙ₊₁(x) = 2xUₙ(x) - Uₙ₋₁(x)
- Orthogonal with respect to the weight function (1-x²)^(1/2)

```python
from pytorch_kan.basis import ChebyshevSecond

basis = ChebyshevSecond(degree=5)
```

### Gegenbauer Polynomials
- Generalization of Chebyshev and Legendre polynomials
- Include an additional parameter (α) to control the weight function

```python
from pytorch_kan.basis import Gegenbauer

basis = Gegenbauer(degree=5, alpha=1.5)
```

### Hermite Polynomials
- Well-suited for functions on unbounded domains
- Orthogonal with respect to a Gaussian weight function

```python
from pytorch_kan.basis import Hermite

basis = Hermite(degree=5)
```

### Jacobi Polynomials
- Generalization with two parameters (α, β)
- Include Chebyshev, Gegenbauer, and Legendre as special cases

```python
from pytorch_kan.basis import Jacobi

basis = Jacobi(degree=5, alpha=1.0, beta=1.0)
```

### Laguerre Polynomials
- Useful for functions on semi-infinite intervals [0, ∞)
- Orthogonal with respect to an exponential weight function

```python
from pytorch_kan.basis import Laguerre

basis = Laguerre(degree=5)
```

### Legendre Polynomials
- Special case of Jacobi polynomials (α = β = 0)
- Orthogonal with respect to a uniform weight function

```python
from pytorch_kan.basis import Legendre

basis = Legendre(degree=5)
```

### Matrix KAN
- Implementation that uses matrix operations instead of explicit polynomial evaluation
- Can offer computational advantages for certain problems

```python
from pytorch_kan.basis import MatrixKAN

basis = MatrixKAN()
```

## Implementation Details

Each basis function class implements the following methods:

- `__init__(degree, **kwargs)`: Initialize with the polynomial degree and any additional parameters
- `forward(x)`: Transform the input tensor using the basis function
- `extra_repr()`: Return a string containing the extra representational information

## Usage with KAN Models

```python
from pytorch_kan.basis import Legendre
from pytorch_kan.nn import KAN

# Create a basis function
basis = Legendre(degree=5)

# Use it in a KAN model
model = KAN(
    input_dim=10,
    output_dim=1,
    basis_func=basis,
    hidden_dim=64
)
```

## Performance Considerations

- Higher-degree polynomials provide more expressiveness but require more computation
- Consider the domain of your data when selecting a basis function
- Some basis functions are better suited for specific types of problems
  - Chebyshev: Good for oscillatory functions
  - Hermite: Good for functions that decay rapidly
  - Laguerre: Good for functions on [0, ∞)
