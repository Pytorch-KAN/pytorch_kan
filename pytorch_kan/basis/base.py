from abc import ABC, abstractmethod
import torch
import plotly.graph_objects as go

class BaseBasis(ABC):
    def __init__(self, order: int):
        self.order = order

    @abstractmethod
    def calculate_basis(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def visualize_polynomial(self, x: torch.Tensor, indices: tuple):
        basis_values = self.calculate_basis(x)
        polynomial_values = basis_values[indices]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x[indices].cpu().numpy(), y=polynomial_values.cpu().numpy(), mode='lines', name=f'Polynomial {indices}'))
        fig.update_layout(title=f'Visualization of Polynomial {indices}', xaxis_title='x', yaxis_title='y')
        fig.show()