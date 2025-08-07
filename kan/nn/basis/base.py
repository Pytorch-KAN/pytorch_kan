"""Base classes and utilities for KAN basis functions.

This module defines :class:`BaseBasis`, an abstract base class that all
specific basis implementations should inherit from.  It provides common
functionalities such as device/dtype migration and simple visualization
helpers.
"""

from abc import ABC, abstractmethod
import warnings

import torch


class BaseBasis(ABC):
    """Abstract base class for all basis families.

    Parameters
    ----------
    order: int
        Maximum order of the basis functions produced by the class.
    device: torch.device, optional
        Device on which any tensor attributes should live.  If ``None`` the
        device will be inferred from tensors passed at runtime.
    dtype: torch.dtype, optional
        Desired data type for tensor attributes.  If ``None`` the dtype of
        existing tensors is preserved.
    """

    def __init__(self, order: int, device=None, dtype=None):
        self.order = order
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute basis representation for an input tensor.

        Sub-classes **must** override this method to produce a tensor whose
        last dimension enumerates the basis functions up to ``self.order``.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor for which the basis representation is calculated.

        Returns
        -------
        torch.Tensor
            Basis tensor whose shape should broadcast with ``x`` and has an
            additional dimension containing ``order`` elements.
        """
        raise NotImplementedError

    def to(self, device=None, dtype=None):
        """Move the basis to the specified device and/or dtype.

        This method recursively moves all tensor attributes (and any nested
        :class:`torch.nn.Module` instances) to the requested device or dtype so
        that basis objects can be transferred between CPUs and GPUs in a
        standardised way.

        Parameters
        ----------
        device: torch.device, optional
            Target device.
        dtype: torch.dtype, optional
            Target data type.

        Returns
        -------
        BaseBasis
            ``self`` for chaining.
        """
        for name, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device=device, dtype=dtype))
            elif isinstance(value, torch.nn.Module):
                value.to(device=device, dtype=dtype)

        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype

        return self

    def visualize_polynomial(self, x: torch.Tensor, indices: tuple[int, int]):
        """Visualize a single polynomial from the calculated basis.

        Parameters
        ----------
        x: torch.Tensor
            Two-dimensional input tensor of shape ``(batch, features)`` used to
            generate the basis values.
        indices: tuple[int, int]
            A pair ``(batch_index, feature_index)`` selecting the sample and
            feature for which the polynomial should be visualised.

        Notes
        -----
        This function relies on :mod:`plotly` for visualisation.  If Plotly is
        not installed, a :class:`UserWarning` is emitted explaining how to
        enable the feature.
        """

        if len(indices) != 2:
            raise ValueError("indices must be a tuple of (batch_index, feature_index)")

        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            warnings.warn(
                "plotly is not installed. Visualization not available. "
                "Install plotly to enable this feature.",
                UserWarning,
            )
            return

        batch_idx, feature_idx = indices
        basis_values = self.calculate_basis(x)
        polynomial_values = basis_values[batch_idx, feature_idx, :]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x[batch_idx, feature_idx].cpu().numpy(),
                y=polynomial_values.cpu().numpy(),
                mode="lines",
                name=f"Polynomial {indices}",
            )
        )
        fig.update_layout(
            title=f"Visualization of Polynomial {indices}",
            xaxis_title="x",
            yaxis_title="y",
        )
        fig.show()
