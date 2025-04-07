import torch
import torch.nn.functional as F
from functools import lru_cache
from pytorch_kan.basis.base import BaseBasis

class BSplineBasis(BaseBasis):
    """
    B-spline basis functions for local control in Kolmogorov-Arnold Networks.
    
    B-splines (basis splines) are piecewise polynomial functions with compact support,
    making them ideal for local control in function approximation. They provide
    smooth transitions between segments and can be efficiently computed using
    recursive formulations.
    
    Attributes:
        order (int): The number of basis functions to generate.
        degree (int): The degree of the B-spline polynomials (e.g., degree=3 for cubic).
        knots (torch.Tensor): Knot vector defining the B-spline's domain partitioning.
    """
    def __init__(self, order: int, degree: int, knots: torch.Tensor):
        """
        Initialize a B-spline basis function.
        
        Args:
            order (int): The number of basis functions to generate.
            degree (int): The degree of the B-spline polynomials.
                          (degree=1: linear, degree=2: quadratic, degree=3: cubic)
            knots (torch.Tensor): Knot vector that defines the domain partitioning.
                                  Should have length (order + degree + 1).
        """
        super().__init__(order)
        self.degree = degree
        self.knots = knots

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the B-spline basis expansion for the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            
        Returns:
            torch.Tensor: B-spline basis values of shape [batch_size, input_dim, order].
        """
        return self.bspline_basis_expansion(x, self.degree, self.knots, self.order)

    @staticmethod
    @lru_cache(maxsize=None)
    def _binomial_coefficients_matrix(spline_degree):
        """
        Compute binomial coefficients matrix for B-spline calculation.
        
        This method uses the lru_cache decorator for memoization to avoid
        recomputing the coefficients for the same spline degree.
        
        Args:
            spline_degree (int): Degree of the B-spline.
            
        Returns:
            torch.Tensor: Matrix of binomial coefficients.
        """
        basis = torch.zeros(spline_degree + 1, spline_degree + 1)
        for i in range(spline_degree + 1):
            for j in range(i + 1):
                coef = 1
                for k in range(1, j + 1):
                    coef *= (i - k + 1)
                    coef //= k
                basis[i, j] = coef * (-1)**(i - j)
        return basis

    @staticmethod
    @lru_cache(maxsize=None)
    def _vandermonde_matrix(spline_degree):
        """
        Compute the Vandermonde matrix for B-spline calculation.
        
        The Vandermonde matrix is used to convert between power basis and
        B-spline basis representations.
        
        Args:
            spline_degree (int): Degree of the B-spline.
            
        Returns:
            torch.Tensor: Inverted Vandermonde matrix.
        """
        t = torch.linspace(0, 1, spline_degree + 1)
        basis_matrix = torch.zeros((spline_degree + 1, spline_degree + 1))
        for i in range(spline_degree + 1):
            basis_matrix[i] = t ** i
        return torch.inverse(basis_matrix)

    @staticmethod
    def cox_de_boor_basis(x, i, k, knots):
        """
        Evaluate B-spline basis function using the Cox-de Boor recursion formula.
        
        This recursive implementation computes the value of the i-th B-spline
        basis function of degree k at point x.
        
        Args:
            x (float): Point at which to evaluate the basis function.
            i (int): Index of the basis function.
            k (int): Degree of the basis function.
            knots (torch.Tensor): Knot vector.
            
        Returns:
            float: Value of the basis function at point x.
        """
        if k == 0:
            return 1.0 if knots[i] <= x < knots[i+1] else 0.0
        d1 = knots[i+k] - knots[i]
        d2 = knots[i+k+1] - knots[i+1]
        f1 = 0.0 if d1 == 0.0 else (x - knots[i]) / d1
        f2 = 0.0 if d2 == 0.0 else (knots[i+k+1] - x) / d2
        return f1 * BSplineBasis.cox_de_boor_basis(x, i, k-1, knots) + f2 * BSplineBasis.cox_de_boor_basis(x, i+1, k-1, knots)

    @staticmethod
    def _de_boor_matrix(x, knots, coefficients, degree):
        """
        Evaluate B-spline using the de Boor algorithm (matrix form).
        
        This is a non-recursive implementation of the de Boor algorithm
        for efficient evaluation of B-splines.
        
        Args:
            x (torch.Tensor): Points at which to evaluate the B-spline.
            knots (torch.Tensor): Knot vector.
            coefficients (torch.Tensor): Control point coefficients.
            degree (int): Degree of the B-spline.
            
        Returns:
            torch.Tensor: Evaluated B-spline values.
        """
        batch_size, input_dim = x.shape
        indices = torch.searchsorted(knots[degree:-degree], x.flatten()) - 1
        indices = torch.clamp(indices, 0, len(knots) - degree - 2)
        d = torch.zeros((batch_size * input_dim, degree + 1), device=x.device)
        for i in range(degree + 1):
            idx = indices + i
            d[:, i] = coefficients[idx]
        for r in range(1, degree + 1):
            for j in range(degree, r - 1, -1):
                alpha = torch.zeros_like(x.flatten())
                left_knot = knots[indices + j - r]
                right_knot = knots[indices + j]
                denominator = right_knot - left_knot
                mask = denominator != 0
                alpha[mask] = (x.flatten()[mask] - left_knot[mask]) / denominator[mask]
                d[:, j] = (1 - alpha) * d[:, j-1] + alpha * d[:, j]
        return d[:, degree].reshape(batch_size, input_dim)

    @staticmethod
    def bspline_basis_expansion(x, degree, knots, num_bases):
        """
        Compute B-spline basis function values for all basis functions.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            degree (int): Degree of the B-spline.
            knots (torch.Tensor): Knot vector.
            num_bases (int): Number of basis functions to compute.
            
        Returns:
            torch.Tensor: B-spline basis values of shape [batch_size, input_dim, num_bases].
        """
        batch_size, input_dim = x.shape
        basis_values = torch.zeros(batch_size, input_dim, num_bases, device=x.device)
        for i in range(num_bases):
            for b in range(batch_size):
                for d in range(input_dim):
                    basis_values[b, d, i] = BSplineBasis.cox_de_boor_basis(x[b, d], i, degree, knots)
        return basis_values

class RBFBasis(BaseBasis):
    """
    Radial Basis Function (RBF) basis for local control in Kolmogorov-Arnold Networks.
    
    RBFs are radially symmetric functions centered at specific points in the input space.
    They provide local approximation capability by activating strongly near their centers
    and decaying with distance. This implementation uses Gaussian RBFs.
    
    Attributes:
        order (int): The number of basis functions to use.
        centers (torch.Tensor): The center points of each RBF.
        sigma (float): The width parameter controlling the spread of each RBF.
    """
    def __init__(self, order: int, centers: torch.Tensor, sigma: float):
        """
        Initialize an RBF basis function set.
        
        Args:
            order (int): The number of basis functions to use.
            centers (torch.Tensor): Center points for the RBFs, shape should be
                                   [num_centers, feature_dim].
            sigma (float): Width parameter controlling the spread of each RBF.
                           Larger values create wider, more overlapping RBFs.
        """
        super().__init__(order)
        self.centers = centers
        self.sigma = sigma

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the RBF basis expansion for the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            
        Returns:
            torch.Tensor: RBF basis values of shape [batch_size, input_dim, num_centers].
        """
        return self.rbf_basis(x, self.centers, self.sigma)

    @staticmethod
    def rbf_basis(x, centers, sigma):
        """
        Compute Gaussian RBF values for all centers.
        
        For each input point and RBF center, computes:
        φ(x) = exp(-||x - center||²/σ²)
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            centers (torch.Tensor): Center points for the RBFs, shape [num_centers, feature_dim].
            sigma (float): Width parameter controlling the spread of each RBF.
            
        Returns:
            torch.Tensor: RBF values of shape [batch_size, input_dim, num_centers].
        """
        batch_size, input_dim = x.shape
        num_rbf_bases = len(centers)
        rbf_values = torch.zeros(batch_size, input_dim, num_rbf_bases, device=x.device)
        for i, center in enumerate(centers):
            squared_dist = torch.sum((x.unsqueeze(2) - center.view(1, 1, -1))**2, dim=-1)
            rbf_values[:, :, i] = torch.exp(-squared_dist / (sigma**2))
        return rbf_values

class SmoothActivationBasis(BaseBasis):
    """
    Smooth activation function basis for local control in Kolmogorov-Arnold Networks.
    
    This class provides a collection of smooth, differentiable activation functions
    that can be used as basis functions in KANs. These functions typically have 
    localized responses and can model complex, non-linear relationships.
    
    Supported activation types:
    - 'silu': Sigmoid Linear Unit (Swish) - x * sigmoid(x)
    - 'gelu': Gaussian Error Linear Unit - x * Φ(x)
    - 'softplus': log(1 + exp(β*x))/β
    - 'mish': x * tanh(softplus(x))
    - 'elu': Exponential Linear Unit - x if x > 0 else α * (exp(x) - 1)
    - 'tanh_shrink': x - tanh(x)
    
    Attributes:
        order (int): The number of basis functions (unused in this implementation).
        activation_type (str): The type of smooth activation function to use.
        alpha (float): Parameter for certain activation functions (e.g., ELU).
        beta (float): Additional parameter for scaling (e.g., softplus).
    """
    def __init__(self, order: int, activation_type: str, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize a smooth activation basis function.
        
        Args:
            order (int): The order parameter (mainly for compatibility with BaseBasis).
            activation_type (str): Type of activation function to use.
                                  Options: 'silu', 'gelu', 'softplus', 'mish', 'elu', 'tanh_shrink'
            alpha (float, optional): Parameter for certain activation functions. Defaults to 1.0.
            beta (float, optional): Additional parameter for certain activations. Defaults to 1.0.
                                   
        Raises:
            ValueError: If an unsupported activation type is specified.
        """
        super().__init__(order)
        self.activation_type = activation_type
        self.alpha = alpha
        self.beta = beta

    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the activation function values for the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            
        Returns:
            torch.Tensor: Activation values with the same shape as the input.
            
        Raises:
            ValueError: If an unsupported activation type is specified.
        """
        if self.activation_type == 'silu':
            return self._silu(x)
        elif self.activation_type == 'gelu':
            return self._gelu(x)
        elif self.activation_type == 'softplus':
            return self._softplus(x)
        elif self.activation_type == 'mish':
            return self._mish(x)
        elif self.activation_type == 'elu':
            return self._elu(x)
        elif self.activation_type == 'tanh_shrink':
            return self._tanh_shrink(x)
        else:
            raise ValueError(f"Unsupported activation type: {self.activation_type}")

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU/Swish activation function: x * sigmoid(x)
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Activated tensor.
        """
        return x * torch.sigmoid(x)

    def _gelu(self, x: torch.Tensor) -> torch.Tensor:
        """
        GELU activation function: x * Φ(x) where Φ is the CDF of the standard normal distribution.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Activated tensor.
        """
        return F.gelu(x)

    def _softplus(self, x: torch.Tensor) -> torch.Tensor:
        """
        Softplus activation function: log(1 + exp(β*x))/β
        
        This implementation uses a threshold to prevent numerical overflow for large inputs.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Activated tensor.
        """
        result = torch.zeros_like(x)
        mask = x > 20
        result[mask] = x[mask]
        not_mask = ~mask
        result[not_mask] = torch.log(1 + torch.exp(self.beta * x[not_mask])) / self.beta
        return result

    def _mish(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mish activation function: x * tanh(softplus(x))
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Activated tensor.
        """
        return x * torch.tanh(F.softplus(x))

    def _elu(self, x: torch.Tensor) -> torch.Tensor:
        """
        ELU activation function: x if x > 0 else α * (exp(x) - 1)
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Activated tensor.
        """
        return F.elu(x, alpha=self.alpha)

    def _tanh_shrink(self, x: torch.Tensor) -> torch.Tensor:
        """
        TanhShrink activation function: x - tanh(x)
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Activated tensor.
        """
        return x - torch.tanh(x)


