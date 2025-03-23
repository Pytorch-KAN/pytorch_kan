import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np

# Import MatrixKAN layer
from src.nn.layers import MatrixKANLayer


class MatrixKANMNISTModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super(MatrixKANMNISTModel, self).__init__()
        # Flatten 28x28 images to a 784-dimensional vector
        self.flatten = nn.Flatten()
        
        # MatrixKAN layer from 784 input dimensions to hidden_dim
        self.kan_layer1 = MatrixKANLayer(
            input_dim=784,
            output_dim=hidden_dim,
            spline_degree=3,
            grid_size=100,  # Standard grid size for MatrixKAN
        )
        
        # Output layer: hidden_dim to 10 classes
        self.kan_layer2 = MatrixKANLayer(
            input_dim=hidden_dim,
            output_dim=10,
            spline_degree=3,  # Use standard spline degree
            grid_size=100,    # Standard grid size for consistent comparison
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


def plot_results(train_losses, test_accuracies, save_path='output/mnist_matrixkan_results.png'):
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
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transformations for MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Create MatrixKAN model
    model = MatrixKANMNISTModel(hidden_dim=256).to(device)
    
    # Print model summary
    print(f"MatrixKAN MNIST model architecture:")
    print(model)
    print(f"Number of trainable parameters: {count_parameters(model):,}")
    
    # Train the model
    train_losses, test_accuracies, training_time = train(
        model, train_loader, test_loader, epochs=10, lr=0.001, device=device
    )
    
    # Plot results
    plot_results(train_losses, test_accuracies, 'output/mnist_matrixkan_results.png')
    
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
        'parameters': count_parameters(model)
    }, 'output/mnist_matrixkan_model.pth')
    print("Model and metrics saved to 'output/mnist_matrixkan_model.pth'")


if __name__ == "__main__":
    main()