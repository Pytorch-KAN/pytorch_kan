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

The package expects a working PyTorch installation appropriate for your platform (CPU, CUDA, or ROCm). Install PyTorch first following guidance from [pytorch.org](https://pytorch.org/).

- CPU-only example:
  ```bash
  pip install torch torchvision torchaudio
  ```
- CUDA example (CUDA 12.1):
  ```bash
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  ```

Then install the library from PyPI:

```bash
pip install pytorch-kan
```

For local development with all dependencies, see the Development section below.

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

## Development

### Local (virtualenv)

Requirements: Python 3.12 recommended.

```bash
git clone https://github.com/yourusername/pytorch_kan.git
cd pytorch_kan
python -m venv .venv
source .venv/bin/activate
# Install all development dependencies and the package in editable mode
pip install -e '.[dev]'

# Run tests
pytest

# Run a tutorial or your own script
python tutorials/main.py
```

### Docker (CPU-only)

Build and run the CPU dev image (uses `python:3.12-slim`, installs `-e '.[dev]'`):

```bash
make build
make container-cpu
```

Inside the container, the project and dev dependencies are already installed. Your local repo is mounted at `/workspace`.

### Docker (GPU / CUDA)

Build and run the CUDA dev image. This installs CUDA-enabled PyTorch wheels (cu121) and then `-e '.[dev]'`.

```bash
make build-cuda
make container-gpu
```

Notes:
- Requires a compatible NVIDIA driver on the host and Docker with `--gpus all` support.
- Adjust the CUDA wheel index URL in `Dockerfile.cuda` if you need a different CUDA version.

### Packaging and Publishing

Build sdist and wheel, and publish to PyPI:

```bash
make dist
make publish  # requires TWINE_USERNAME/TWINE_PASSWORD or token configured
```

## Repository Structure

- `kan`: Python package containing KAN layers, basis functions and utilities.
- `tests`: PyTest-based unit tests for basis functions and layers.
- `tutorials`: Example scripts demonstrating KANs on tasks such as MNIST.
- `requirements.txt`: Minimal runtime pins (dev dependencies are managed via `.[dev]`).

## Running Tests

After making changes, run the test suite:

```bash
pytest
```

## License

This repository is released under the MIT License.  See `LICENSE` for details.
