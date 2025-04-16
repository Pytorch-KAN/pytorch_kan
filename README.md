# PyTorch KAN (Kolmogorov-Arnold Networks)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

PyTorch KAN is an efficient and scalable implementation of Kolmogorov-Arnold Networks, a neural network architecture based on the Kolmogorov-Arnold representation theorem. This library provides a flexible framework for building and training KANs with various basis functions.

## Installation

You can install PyTorch KAN in several ways:

### From PyPI (Recommended)
```bash
# Basic installation
pip install pytorch-kan

# Install with all optional dependencies
pip install pytorch-kan[all]

# Install specific extras
pip install pytorch-kan[visualization]  # For plotting utilities
pip install pytorch-kan[notebook]       # For Jupyter notebook support
pip install pytorch-kan[transformers]   # For transformer model support
```

### From Source

Clone the repository and install locally:
```bash
git clone https://github.com/username/pytorch_kan.git
cd pytorch_kan

# Using pip
pip install -e .                # Basic installation
pip install -e ".[all]"        # Install with all optional dependencies

# Using Poetry (recommended for development)
poetry install                  # Install with all dependencies
poetry install --no-dev        # Install without development dependencies
```

### Using Docker

A Docker container with GPU support is available:
```bash
# Build the container
make build

# Run the container
make container
```

## Key Features

- **Multiple Basis Functions**: Support for Chebyshev, Gegenbauer, Hermite, Jacobi, Laguerre, Legendre, and Matrix KAN implementations
- **High Performance**: Optimized for efficiency and scalability
- **PyTorch Integration**: Seamless integration with the PyTorch ecosystem
- **Comprehensive Examples**: MNIST classification tutorials for each basis type

## Project Structure

- `src/pytorch_kan/`: Core implementation
  - `basis/`: Basis function implementations (Chebyshev, Gegenbauer, etc.)
  - `nn/`: Neural network modules for building KANs
- `tutorials/`: Example implementations
  - `mnist/`: MNIST classification examples using different basis functions
- `data/`: Dataset storage
- `output/`: Model outputs and visualization results

## Quick Start

```python
import torch
from pytorch_kan.nn import KAN
from pytorch_kan.basis import ChebyshevFirst

# Define a simple KAN model
model = KAN(
    input_dim=784,
    output_dim=10,
    basis_func=ChebyshevFirst(),
    hidden_dim=128
)

# Use like any PyTorch model
x = torch.randn(32, 784)
output = model(x)
```

## Examples

Check the `tutorials/mnist/` directory for complete examples using different basis functions:

```bash
# Run an example
python tutorials/mnist/chebyshev_first.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
