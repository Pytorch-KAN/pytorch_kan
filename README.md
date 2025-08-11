# PyTorch KAN

Efficient and scalable implementation of Kolmogorov-Arnold Networks (KANs).

## Philosophy

PyTorch KAN explores neural networks built from basis functions.  Layers are
constructed using either **local control** (e.g. B-splines or radial basis
functions) or **global control** (e.g. Fourier or wavelet expansions).  The
library mirrors the familiar `torch.nn` API so that experimenting with KANs
feels like using standard PyTorch modules.

The project emphasises:

- modular design for easy swapping of basis functions and layers,
- clear separation between local and global control components,
- compatibility with PyTorch tensors and training utilities.

## Installation

The library can be installed directly from PyPI:

```bash
pip install pytorch_kan
```

For local development, clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/pytorch_kan.git
cd pytorch_kan
pip install .
```

The package expects a working PyTorch installation.  You can install PyTorch
from [pytorch.org](https://pytorch.org/).

## Getting Started

```python
from kan import nn
import torch

# Local control layer using B-spline basis
layer = nn.MatrixKANLayer(input_dim=4, output_dim=2)
x = torch.randn(1, 4)
output = layer(x)

# Global control layer using Fourier basis
fourier = nn.global_layers.FourierKANLayer(input_dim=4, output_dim=2, order=3)
output2 = fourier(x)
```

`local_layers`, `local_basis`, `global_layers` and `global_basis` are available
under `kan.nn` for fine-grained control.

## Repository Structure

- `kan`: Python package containing KAN layers, basis functions and utilities.
- `tests`: PyTest-based unit tests for basis functions and layers.
- `tutorials`: Example scripts demonstrating KANs on tasks such as MNIST.
- `requirements.txt`: Python dependencies for development.

## Running Tests

After making changes, run the test suite:

```bash
pytest
```

## License

This repository is released under the MIT License.  See `LICENSE` for details.
