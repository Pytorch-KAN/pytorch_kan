import torch
from ...basis.base import BaseBasis


class RBFBasis(BaseBasis):
    """Per-dimension Gaussian RBFs: centers shape ``(input_dim, num_centers)``."""

    def __init__(self, order: int, centers: torch.Tensor):
        """Create an RBF basis with per-dimension centres.

        Parameters
        ----------
        order:
            Number of basis functions per input dimension.
        centers:
            Tensor containing the centres of shape ``(input_dim, order)``.
        """

        super().__init__(order)
        self.centers = centers  # (input_dim, order)

    def calculate_basis(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Evaluate Gaussian RBFs.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, input_dim)``.
        sigma:
            Positive width parameter for the Gaussian kernels.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch, input_dim, order)`` containing the
            evaluated RBF basis functions.
        """

        # x: (B, I), centers: (I, C) -> diff: (B, I, C)
        diff = x.unsqueeze(-1) - self.centers.unsqueeze(0)
        sq = diff * diff
        return torch.exp(-sq / (sigma * sigma))

