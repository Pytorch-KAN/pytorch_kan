import torch
from ...basis.base import BaseBasis


class RBFBasis(BaseBasis):
    """Radial basis function (RBF) family for local control modules."""

    def __init__(self, order: int, centers: torch.Tensor, sigma: float):
        """Parameters
        ----------
        order: int
            Number of basis functions.
        centers: torch.Tensor
            Tensor of shape ``(num_bases, input_dim)`` representing RBF centers.
        sigma: float
            Width parameter of the Gaussian kernels.
        """
        super().__init__(order)
        self.centers = centers
        self.sigma = sigma

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RBF basis values for ``x``."""
        return self.rbf_basis(x, self.centers, self.sigma)

    @staticmethod
    def rbf_basis(x, centers, sigma):
        """Evaluate Gaussian RBFs for each center."""
        batch_size, input_dim = x.shape
        num_rbf_bases = len(centers)
        rbf_values = torch.zeros(batch_size, input_dim, num_rbf_bases, device=x.device)
        for i, center in enumerate(centers):
            squared_dist = torch.sum((x.unsqueeze(2) - center.view(1, 1, -1)) ** 2, dim=-1)
            rbf_values[:, :, i] = torch.exp(-squared_dist / (sigma ** 2))
        return rbf_values
