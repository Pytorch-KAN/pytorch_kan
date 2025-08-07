# Local Control

Components for KAN layers with local receptive fields.  These modules employ
bases such as B-splines or radial basis functions to provide fine-grained
control over subsets of the input.

Subpackages:

- `basis`: local basis families (B-splines, RBFs, smooth activations).
- `layers`: layer implementations built on top of local bases.
