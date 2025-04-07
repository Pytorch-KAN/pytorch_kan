# MNIST Classification with KAN

This tutorial demonstrates how to use Kolmogorov-Arnold Networks (KAN) for image classification using the MNIST dataset.

## Introduction

The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9) and is a common benchmark for testing machine learning algorithms. In this tutorial, we'll build a KAN model to classify these images.

## Prerequisites

Before starting this tutorial, make sure you have:

1. Installed PyTorch KAN and its dependencies
2. Basic understanding of PyTorch
3. Familiarity with the MNIST dataset

## Loading the MNIST Dataset

We'll start by loading the MNIST dataset using PyTorch's built-in datasets:

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load training and test datasets
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
```

## Building a KAN Model

Next, we'll create a KAN model for classification:

```python
import torch.nn as nn
import torch.nn.functional as F
from pytorch_kan.nn import KAN
from pytorch_kan.basis import ChebyshevFirst

class KANNetwork(nn.Module):
    def __init__(self):
        super(KANNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.kan = KAN(
            in_features=784,  # 28x28 input image
            out_features=10,  # 10 output classes (digits 0-9)
            hidden_features=[128, 64],
            basis_func=ChebyshevFirst(n_funcs=5)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.kan(x)

# Create the model
model = KANNetwork()
```

## Training the Model

Now, let's train the model on the MNIST dataset:

```python
import torch.optim as optim

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        running_loss += loss.item()
        
        if batch_idx % 100 == 99:
            print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
            running_loss = 0.0
```

## Evaluating the Model

Finally, let's evaluate our trained model:

```python
# Test the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
```

## Visualizing Results

We can visualize some predictions made by our model:

```python
import matplotlib.pyplot as plt
import numpy as np

# Load a batch of test data
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Get model predictions
images = images.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)
predicted = predicted.cpu().numpy()

# Plot some images with their predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i in range(10):
    axes[i].imshow(images[i].cpu().squeeze().numpy(), cmap='gray')
    axes[i].set_title(f'Pred: {predicted[i]}, True: {labels[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

## Conclusion

In this tutorial, we've seen how to use Kolmogorov-Arnold Networks for image classification on the MNIST dataset. KANs provide an alternative to traditional neural networks with potentially better interpretability and efficiency.

You can experiment with different basis functions, network architectures, and hyperparameters to improve performance further.