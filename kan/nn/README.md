# kan.nn

Neural network components for building Kolmogorov-Arnold Networks.  The module
organises layers and basis functions into local and global control families and
re-exports convenient aliases for quick access.

Typical usage:

```python
from kan import nn
layer = nn.MatrixKANLayer(input_dim=4, output_dim=2)
fourier = nn.global_layers.FourierKANLayer(input_dim=4, output_dim=2, order=3)
```

Subpackages:

- `basis`: base classes and utilities shared by basis implementations.
- `local_control`: layers and bases with local support such as B-splines.
- `global_control`: layers and bases with global support such as Fourier or wavelet expansions.
