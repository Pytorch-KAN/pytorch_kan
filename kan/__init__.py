"""Top-level package for the PyTorch KAN library."""

from . import nn

__all__ = ["nn"]

# Re-export a package version so that downstream libraries can check
# for compatibility if needed.  The version string should match the
# ``version`` field declared in ``pyproject.toml``.
__version__ = "0.1.0"
