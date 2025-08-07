from .local_control.layers import MatrixKANLayer
from . import local_control, global_control

# Convenient re-exports
local_basis = local_control.basis
local_layers = local_control.layers

global_basis = global_control.basis
global_layers = global_control.layers

__all__ = [
    "MatrixKANLayer",
    "local_basis",
    "local_layers",
    "global_basis",
    "global_layers",
]
