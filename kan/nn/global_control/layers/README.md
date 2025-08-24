# Global Control Layers

Layers using global basis expansions to transform inputs.  Available
implementations include `FourierKANLayer`, `OrthogonalKANLayer`,
`SmoothActivationKANLayer` and `WaveletKANLayer`.

Example:

```python
from kan import nn
layer = nn.global_layers.FourierKANLayer(input_dim=4, output_dim=2, order=3)
out = layer(x)
```
