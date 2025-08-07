# pytorch_kan

Efficient and scalable implementation of Kolmogorov-Arnold Networks.

## Usage
The library follows a PyTorch-like structure. Import the neural network
submodule and choose between local and global control layers or bases:

```python
from kan import nn
layer = nn.MatrixKANLayer(input_dim=4, output_dim=2)
```

Local and global control components are organized under
`kan.nn.local_control` and `kan.nn.global_control` respectively.
