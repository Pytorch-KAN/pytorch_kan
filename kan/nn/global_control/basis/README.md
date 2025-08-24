# Global Basis Functions

Orthogonal basis families used by global control layers.  Example:

```python
from kan.nn.global_control.basis import FourierBasis
basis = FourierBasis(order=3)
values = basis.calculate_basis(x)
```
