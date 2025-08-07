from abc import ABC, abstractmethod
import torch


class BaseBasis(ABC):
    def __init__(self, order: int):
        self.order = order

    @abstractmethod
    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute basis representation for input tensor."""
        raise NotImplementedError

    def visualize_polynomial(self, x: torch.Tensor, indices: tuple):
        """
        Visualize the polynomial at the given indices.

        Note: This requires plotly to be installed.
        If plotly is not available, a warning message is displayed.
        You can install plotly in a virtual environment:

            python3 -m venv ~/kan_venv
            source ~/kan_venv/bin/activate
            pip install plotly

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
        fig.add_trace(
            go.Scatter(
                x=x[indices].cpu().numpy(),
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
