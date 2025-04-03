# Installation

## Using pip

The simplest way to install PyTorch KAN is via pip:

```bash
pip install pytorch-kan
```

## From Source

For the latest development version, you can install directly from the GitHub repository:

```bash
git clone https://github.com/username/pytorch_kan.git
cd pytorch_kan
pip install -e .
```

## Dependencies

PyTorch KAN requires the following packages:

- Python >=3.8
- PyTorch >=2.0.0
- NumPy >=1.24.0
- SciPy >=1.10.0
- scikit-learn >=1.2.0

Optional dependencies for visualization and examples:

- matplotlib >=3.6.0
- seaborn >=0.12.0
- plotly >=5.13.0

## Installation with Optional Dependencies

You can install PyTorch KAN with optional dependencies using the extras:

```bash
# For visualization tools
pip install pytorch-kan[visualization]

# For notebook examples
pip install pytorch-kan[notebook]

# For transformer examples
pip install pytorch-kan[transformers]

# For all optional dependencies
pip install pytorch-kan[all]
```