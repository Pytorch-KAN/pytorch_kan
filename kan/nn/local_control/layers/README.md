# Local Control Layers

KAN layer implementations built on local basis functions.  Provided layers
include `MatrixKANLayer`, `RBFKANLayer` and `SmoothActivationKANLayer`.

Example:

```python
from kan import nn
layer = nn.MatrixKANLayer(input_dim=4, output_dim=2)
```
