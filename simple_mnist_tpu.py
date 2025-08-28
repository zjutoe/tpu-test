#!/usr/bin/env python3
"""
Simple MNIST Training on Google TPU with PyTorch/XLA
This example demonstrates basic TPU training using the latest 2025 patterns.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import time
from tqdm import tqdm


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
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


def get_data_loaders(batch_size=128):
    """Create data loaders for MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch using TPU-optimized patterns"""
    model.train()
    
    # Wrap data loader for XLA
    para_loader = pl.ParallelLoader(train_loader, [device])
    train_loader = para_loader.per_device_loader(device)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
        with torch_xla.step():
            # Data is already on XLA device via ParallelLoader
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            # Mark step to ensure metrics are updated
            xm.mark_step()
            accuracy = 100 * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            print(f'Batch {batch_idx}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Final synchronization
    xm.mark_step()
    final_accuracy = 100 * correct / total
    final_loss = running_loss / len(train_loader)
    
    return final_loss, final_accuracy


def test_model(model, device, test_loader):
    """Test model accuracy"""
    model.eval()
    
    # Wrap data loader for XLA
    para_loader = pl.ParallelLoader(test_loader, [device])
    test_loader = para_loader.per_device_loader(device)
    
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            with torch_xla.step():
                output = model(data)
                test_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
    
    # Synchronize before final calculation
    xm.mark_step()
    
    test_loss /= total
    accuracy = 100.0 * correct / total
    
    return test_loss, accuracy


def main():
    """Main training function"""
    print("🚀 Starting MNIST Training on TPU with PyTorch/XLA")
    
    # Configuration
    EPOCHS = 5
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    
    # Get XLA device (TPU)
    device = xm.xla_device()
    print(f"Using device: {device}")
    print(f"Device type: {xm.xla_device_hw(device)}")
    
    # Create model and move to TPU
    model = SimpleCNN(num_classes=10).to(device)
    print("✅ Model created and moved to TPU")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Data loaders
    print("📥 Loading MNIST dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)
    print(f"✅ Dataset loaded - Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Training loop
    print(f"\n🎯 Training for {EPOCHS} epochs...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        # Test
        test_loss, test_acc = test_model(model, device, test_loader)
        
        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # Print XLA metrics for performance monitoring
        print(f"  XLA Metrics: {met.short_metrics_report()}")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time:.2f} seconds")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    
    # Save model
    model_path = 'mnist_tpu_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to {model_path}")


if __name__ == "__main__":
    main()