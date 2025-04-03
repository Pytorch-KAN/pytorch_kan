from abc import ABC, abstractmethod
import torch

class BaseBasis(ABC):
    """
    Abstract base class for all basis functions used in Kolmogorov-Arnold Networks.
    
    This class defines the interface for basis functions that transform input tensors
    into higher-dimensional representations according to the Kolmogorov-Arnold
    representation theorem. All concrete basis function implementations must
    inherit from this class and implement the calculate_basis method.
    
    Attributes:
        order (int): The order or degree of the basis function, controlling 
                     the expressiveness of the approximation.
    """
    def __init__(self, order: int):
        """
        Initialize a basis function with the specified order.
        
        Args:
            order (int): The order or degree of the basis function. Higher orders
                         generally provide more expressive power but require more
                         computation and parameters.
        """
        self.order = order

    @abstractmethod
    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the basis function expansion for the input tensor.
        
        This abstract method must be implemented by all subclasses to compute
        the basis function values for each element in the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim].
            
        Returns:
            torch.Tensor: Transformed tensor containing basis function values,
                         typically of shape [batch_size, input_dim, num_basis_functions]
                         where num_basis_functions depends on the specific implementation.
        """
        pass

    def visualize_polynomial(self, x: torch.Tensor, indices: tuple):
        """
        Visualizes the polynomial basis function at the given indices.
        
        Creates an interactive plot showing the values of the specified polynomial
        basis function across the input domain.
        
        Args:
            x (torch.Tensor): Input tensor containing the domain points to evaluate.
            indices (tuple): Indices specifying which polynomial to visualize.
        
        Note:
            This requires plotly to be installed. If plotly is not available,
            a warning message is displayed with installation instructions.
            
            You can install plotly in a virtual environment:
            
            ```
            python3 -m venv ~/kan_venv
            source ~/kan_venv/bin/activate
            pip install plotly
            ```
            
            Then run your code from within that virtual environment.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Warning: plotly is not installed. Visualization not available.")
            print("To use visualization features, create a virtual environment:")
            print("  python3 -m venv ~/kan_venv")
            print("  source ~/kan_venv/bin/activate")
            print("  pip install plotly")
            print("Then run your code from within that environment.")
            return
            
        # Get basis values and extract the specific polynomial values
        basis_values = self.calculate_basis(x)
        polynomial_values = basis_values[indices]

        # Create the visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x[indices].cpu().numpy(), y=polynomial_values.cpu().numpy(), 
                                mode='lines', name=f'Polynomial {indices}'))
        fig.update_layout(title=f'Visualization of Polynomial {indices}', 
                         xaxis_title='x', yaxis_title='y')
        fig.show()
        
    # def get_polynomial_values(self, x: torch.Tensor, indices: tuple):
    #     """
    #     Returns the polynomial values at the given indices without visualization.
    #     This method doesn't require any additional dependencies.
        
    #     Args:
    #         x: Input tensor
    #         indices: Tuple of indices to extract the specific polynomial
            
    #     Returns:
    #         Tensor of polynomial values
    #     """
    #     basis_values = self.calculate_basis(x)
    #     return basis_values[indices]