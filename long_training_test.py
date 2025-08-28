#!/usr/bin/env python3
"""
Longer Training Test - For observing TPU usage patterns
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
import torch_xla.debug.metrics as met
import time


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


def main():
    print("🚀 Long Training Test - Monitor TPU Usage")
    print("This will run for ~2 minutes to observe TPU patterns")
    
    # Get XLA device
    device = xm.xla_device()
    print(f"Using device: {device}")
    
    # Data setup - larger dataset for longer training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    # Use larger subset
    subset_indices = list(range(0, 10000))  # 10K samples
    train_subset = torch.utils.data.Subset(train_dataset, subset_indices)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
    
    # Create model
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {len(train_subset)} samples for 5 epochs...")
    print("🔍 Check your TPU monitor now!")
    
    start_time = time.time()
    
    # Clear XLA metrics
    met.clear_metrics()
    
    # Training loop
    for epoch in range(5):
        print(f"\n--- Starting Epoch {epoch+1}/5 ---")
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
            
            # Print progress more frequently
            if batch_idx % 25 == 0:
                torch_xla.sync()
                current_acc = 100 * correct / total if total > 0 else 0
                print(f"  Epoch {epoch+1}, Batch {batch_idx:3d}/313, "
                      f"Loss: {loss.item():.4f}, Acc: {current_acc:.1f}%")
        
        torch_xla.sync()
        epoch_acc = 100 * correct / total
        epoch_loss = total_loss / len(train_loader)
        
        print(f"✅ Epoch {epoch+1} complete - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
        # Show XLA metrics every epoch
        print(f"📊 XLA Metrics: {met.short_metrics_report()}")
        
        # Small pause to make monitoring easier
        time.sleep(2)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Long training completed in {total_time:.2f} seconds")
    
    # Final comprehensive metrics
    print("\n📈 Final XLA Performance Report:")
    print(met.metrics_report())
    
    # Save model
    torch.save(model.state_dict(), 'long_training_model.pth')
    print("💾 Model saved as long_training_model.pth")


if __name__ == "__main__":
    main()