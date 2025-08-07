# Local Basis Functions

Basis families with local support used by local-control layers.
Available implementations include `BSplineBasis`, `RBFBasis` and
`SmoothActivationBasis`.

Example:

```python
from kan.nn.local_control.basis import BSplineBasis
basis = BSplineBasis(order=3)
```
