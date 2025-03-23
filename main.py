import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_regression
from src.basis.locals import *
from src.nn.layers import KANLayer

# Generate input tensor: 1000 samples, 10 features
X_train = torch.rand(1000, 10)

# Generate target tensor: 1000 samples, 1 target value
y_train = torch.rand(1000, 1)

# Define the KANLayer
input_dim = 10
output_dim = 1
kan_layer = KANLayer(input_dim, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(kan_layer.parameters(), lr=0.01)

# Training loop
num_epochs = 100
batch_size = 32

for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        # Forward pass
        outputs = kan_layer(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the trained model
with torch.no_grad():
    test_output = kan_layer(X_train)
    test_loss = criterion(test_output, y_train)
    print(f'Final Test Loss: {test_loss.item():.4f}')