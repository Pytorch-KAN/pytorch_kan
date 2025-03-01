import torch
import torch.nn as nn
import torch.optim as optim

class MatrixKAN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, spline_degree=3, grid_size=10, grid_epsilon=0.1):
        super(MatrixKAN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.spline_degree = spline_degree
        self.grid_size = grid_size
        self.grid_epsilon = grid_epsilon
        
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            if i == 0:
                self.layers.append(self._create_layer(input_dim, hidden_layers[i]))
            else:
                self.layers.append(self._create_layer(hidden_layers[i-1], hidden_layers[i]))
        self.layers.append(self._create_layer(hidden_layers[-1], output_dim))
        
        self.register_buffer('basis_matrix', self._create_basis_matrix())

    def _create_layer(self, input_size, output_size):
        return nn.Parameter(torch.randn(input_size, output_size, self.grid_size))

    def _create_basis_matrix(self):
        t = torch.linspace(0, 1, self.spline_degree + 1)
        basis_matrix = torch.zeros((self.spline_degree + 1, self.spline_degree + 1))
        for i in range(self.spline_degree + 1):
            basis_matrix[i] = t ** i
        return torch.inverse(basis_matrix)

    def _calculate_power_bases(self, x):
        x = torch.clamp(x, 0, 1 - self.grid_epsilon)
        indices = torch.floor(x * self.grid_size).long()
        t = (x * self.grid_size) - indices
        power_bases = torch.stack([t**i for i in range(self.spline_degree + 1)], dim=-1)
        return indices, power_bases

    def forward(self, x):
        for layer in self.layers:
            indices, power_bases = self._calculate_power_bases(x)
            spline_outputs = torch.einsum('bij,jk,bik->bi', power_bases, self.basis_matrix, layer[..., indices])
            x = spline_outputs
        return x

    def train_model(self, X, y, epochs=1000, lr=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Example usage
if __name__ == "__main__":
    # Generate some example data
    X = torch.rand(100, 1)
    y = torch.sin(2 * torch.pi * X).squeeze()

    # Create and train the model
    model = MatrixKAN(input_dim=1, output_dim=1, hidden_layers=[10, 10], spline_degree=3)
    model.train_model(X, y)

    # Make predictions
    X_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    with torch.no_grad():
        y_pred = model(X_test)

    print("Training complete. Use model(X) for predictions.")
