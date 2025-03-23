import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np
from src.basis.globals import OrthogonalPolynomial

class LaguerreKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, order=3, grid_epsilon=1e-6):
        super(LaguerreKANLayer, self).__init__()
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
        
        # Create Laguerre polynomial calculator
        self.laguerre_calc = OrthogonalPolynomial(polynomial="laguerre", order=order)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Normalize input
        x = self.norm(x)
        
        # Scale input to [0, 5] range (typical for Laguerre)
        x = torch.clamp(x + 2.5, 0.0 + self.grid_epsilon, 5.0 - self.grid_epsilon)
        
        # Compute Laguerre polynomial basis values
        basis_values = self.laguerre_calc.calculate_basis(x)
        
        # Vectorized computation using einsum
        # Shape: [batch_size, output_dim]
        output = torch.einsum('bip,iop->bo', basis_values, self.weights)
        
        return output

class LaguerreKANMNISTModel(nn.Module):
    def __init__(self, hidden_dim=256, order=5):
        super(LaguerreKANMNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        
        self.kan_layer1 = LaguerreKANLayer(
            input_dim=784, 
            output_dim=hidden_dim, 
            order=order
        )
        
        self.kan_layer2 = LaguerreKANLayer(
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    train_losses = []
    test_accuracies = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        test_accuracy = evaluate(model, test_loader, device)
        test_accuracies.append(test_accuracy)
        
        scheduler.step(epoch_loss)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Loss: {epoch_loss:.4f}, '
              f'Test Accuracy: {test_accuracy:.4f}')
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f'Training completed in {training_time:.2f} seconds')
    
    return train_losses, test_accuracies, training_time

def evaluate(model, test_loader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy

def plot_results(train_losses, test_accuracies, save_path='output/mnist_laguerre_results.png'):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    print(f"Final test accuracy: {test_accuracies[-1]:.4f}")
    print(f"Best test accuracy: {max(test_accuracies):.4f}")
    print(f"Results saved to {save_path}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    order = 5
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    model = LaguerreKANMNISTModel(hidden_dim=256, order=order).to(device)
    
    print(f"Laguerre KAN MNIST model architecture (order = {order}):")
    print(model)
    print(f"Number of trainable parameters: {count_parameters(model):,}")
    
    train_losses, test_accuracies, training_time = train(
        model, train_loader, test_loader, epochs=10, lr=0.001, device=device
    )
    
    plot_results(train_losses, test_accuracies)
    
    final_accuracy = evaluate(model, test_loader, device)
    print(f'Final Test Accuracy: {final_accuracy:.4f}')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_accuracy': final_accuracy,
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'training_time': training_time,
        'parameters': count_parameters(model),
        'order': order
    }, 'output/mnist_laguerre_model.pth')
    print("Model and metrics saved to 'output/mnist_laguerre_model.pth'")

if __name__ == "__main__":
    main()