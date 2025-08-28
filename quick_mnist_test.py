#!/usr/bin/env python3
"""
Quick MNIST Test - Minimal training for demonstration
"""

import os
os.environ['PJRT_DEVICE'] = 'CPU'
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch_xla
import torch_xla.core.xla_model as xm
import time


class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        self.conv = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 12 * 12, 64)
        self.fc2 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    print("🚀 Quick MNIST Test with XLA")
    
    # Get XLA device
    device = xm.xla_device()
    print(f"Using device: {device}")
    
    # Simple data setup (smaller subset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    # Use smaller subset for quick test
    subset_indices = list(range(0, 1000))  # Only 1000 samples
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    
    # Create model
    model = MiniCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {len(train_subset)} samples for 2 epochs...")
    
    start_time = time.time()
    
    # Training loop (very simplified)
    for epoch in range(2):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            with torch_xla.step():
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                torch_xla.sync()
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        torch_xla.sync()
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.2f} seconds")
    print(f"Final accuracy: {accuracy:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'quick_mnist_model.pth')
    print("💾 Model saved as quick_mnist_model.pth")


if __name__ == "__main__":
    main()