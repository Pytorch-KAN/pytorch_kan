import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
from src.basis.globals import OrthogonalPolynomial


class LegendreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, order=3, grid_epsilon=1e-6):
        super(LegendreKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.order = order
        self.grid_epsilon = grid_epsilon
        
        # Learnable weights for combining polynomial outputs
        self.weights = nn.Parameter(
            torch.empty(input_dim, output_dim, order + 1)
        )
        nn.init.kaiming_uniform_(self.weights, mode='fan_in', nonlinearity='relu')
        
        # Layer normalization for input standardization
        self.norm = nn.LayerNorm(input_dim)
        
        # Create Legendre polynomial calculator
        self.legendre_calc = OrthogonalPolynomial(polynomial="legendre", order=order)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Normalize input
        x = self.norm(x)
        
        # Scale input to [-1, 1] range (Legendre domain)
        x = torch.clamp(x, -1.0 + self.grid_epsilon, 1.0 - self.grid_epsilon)
        
        # Compute Legendre polynomial basis values using the OrthogonalPolynomial implementation
        # Shape: [batch_size, input_dim, order+1]
        basis_values = self.legendre_calc.calculate_basis(x)
        
        # Vectorized computation using einsum
        # Multiply each basis value by its corresponding weight and sum
        # Shape: [batch_size, output_dim]
        output = torch.einsum('bip,iop->bo', basis_values, self.weights)
        
        return output


class LegendreKANMNISTModel(nn.Module):
    def __init__(self, hidden_dim=256, order=5):
        super(LegendreKANMNISTModel, self).__init__()
        # Flatten 28x28 images to a 784-dimensional vector
        self.flatten = nn.Flatten()
        
        # First Legendre KAN layer: 784 -> hidden_dim
        self.kan_layer1 = LegendreKANLayer(
            input_dim=784, 
            output_dim=hidden_dim, 
            order=order
        )
        
        # Second Legendre KAN layer: hidden_dim -> 10 classes
        self.kan_layer2 = LegendreKANLayer(
            input_dim=hidden_dim, 
            output_dim=10, 
            order=order
        )
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.kan_layer1(x)
        x = self.kan_layer2(x)
        return x


def train(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Training metrics
    train_losses = []
    test_accuracies = []
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Evaluate on test set
        test_accuracy = evaluate(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        
        # Update learning rate based on validation performance
        scheduler.step(epoch_loss)
        
        # Print epoch statistics
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Loss: {epoch_loss:.4f}, '
              f'Test Accuracy: {test_accuracy:.4f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    # Return training history
    return train_losses, test_accuracies, training_time


def evaluate(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions
            _, predicted = torch.max(output.data, 1)
            
            # Update statistics
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy


def plot_results(train_losses, test_accuracies, save_path='output/mnist_legendre_results.png'):
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Print summary
    print(f"Final test accuracy: {test_accuracies[-1]:.4f}")
    print(f"Best test accuracy: {max(test_accuracies):.4f}")
    print(f"Results saved to {save_path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Enable anomaly detection to find in-place operation issues
    torch.autograd.set_detect_anomaly(True)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set Legendre polynomial order
    order = 5  # Higher order for better expressivity
    
    # Define transformations for MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=80, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Create Legendre KAN model
    model = LegendreKANMNISTModel(hidden_dim=256, order=order).to(device)
    
    # Print model summary
    print(f"Legendre KAN MNIST model architecture (polynomial order = {order}):")
    print(model)
    print(f"Number of trainable parameters: {count_parameters(model):,}")
    
    # Train the model
    train_losses, test_accuracies, training_time = train(
        model, train_loader, test_loader, epochs=10, lr=0.001, device=device
    )
    
    # Plot results
    plot_results(train_losses, test_accuracies, 'output/mnist_legendre_results.png')
    
    # Final evaluation
    final_accuracy = evaluate(model, test_loader, device)
    print(f'Final Test Accuracy: {final_accuracy:.4f}')
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': final_accuracy,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'training_time': training_time,
        'parameters': count_parameters(model),
        'order': order
    }, 'output/mnist_legendre_model.pth')
    print("Model and metrics saved to 'output/mnist_legendre_model.pth'")


if __name__ == "__main__":
    main()