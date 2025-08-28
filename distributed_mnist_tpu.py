#!/usr/bin/env python3
"""
Distributed MNIST Training on Google TPU with PyTorch/XLA
Demonstrates multi-TPU training using DistributedDataParallel and XLA backend.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_backend
import torch_xla.distributed.parallel_loader as pl
import torch.distributed as dist
import time
import os


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


def get_data_loaders(batch_size=128, rank=0, world_size=1):
    """Create distributed data loaders for MNIST dataset"""
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
    
    # Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader, train_sampler


def train_epoch(model, device, train_loader, optimizer, criterion, epoch, rank):
    """Train for one epoch using distributed TPU training"""
    model.train()
    
    # Wrap data loader for XLA
    para_loader = pl.ParallelLoader(train_loader, [device])
    train_loader = para_loader.per_device_loader(device)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        with torch_xla.step():
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
        
        # Print progress every 100 batches (only from rank 0)
        if batch_idx % 100 == 0 and rank == 0:
            xm.mark_step()
            accuracy = 100 * correct / total if total > 0 else 0
            avg_loss = running_loss / (batch_idx + 1)
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Synchronize all processes
    xm.mark_step()
    
    # Calculate final metrics
    final_accuracy = 100 * correct / total if total > 0 else 0
    final_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
    
    return final_loss, final_accuracy


def test_model(model, device, test_loader, rank):
    """Test model accuracy in distributed setting"""
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
    
    # Aggregate results across all processes
    test_loss_tensor = torch.tensor(test_loss).to(device)
    correct_tensor = torch.tensor(correct).to(device)
    total_tensor = torch.tensor(total).to(device)
    
    # All-reduce to get global metrics
    xm.all_reduce(xm.REDUCE_SUM, [test_loss_tensor, correct_tensor, total_tensor])
    
    global_test_loss = test_loss_tensor.item() / total_tensor.item()
    global_accuracy = 100.0 * correct_tensor.item() / total_tensor.item()
    
    return global_test_loss, global_accuracy


def _mp_fn(rank, world_size):
    """Multi-processing function for distributed training"""
    print(f"Process {rank} of {world_size} starting...")
    
    # Initialize distributed training
    dist.init_process_group("xla", rank=rank, world_size=world_size)
    
    # Configuration
    EPOCHS = 5
    BATCH_SIZE = 32  # Per-device batch size
    LEARNING_RATE = 0.01
    
    # Get XLA device
    device = xm.xla_device()
    
    if rank == 0:
        print(f"Using {world_size} TPU cores")
        print(f"Device: {device}")
        print(f"Device type: {xm.xla_device_hw(device)}")
    
    # Create model and move to TPU
    model = SimpleCNN(num_classes=10).to(device)
    
    # Wrap model with DDP
    ddp_model = DDP(model, gradient_as_bucket_view=True)
    
    if rank == 0:
        print("✅ Model created and wrapped with DDP")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=LEARNING_RATE)
    
    # Data loaders with distributed sampling
    if rank == 0:
        print("📥 Loading MNIST dataset...")
    
    train_loader, test_loader, train_sampler = get_data_loaders(
        batch_size=BATCH_SIZE, rank=rank, world_size=world_size
    )
    
    if rank == 0:
        print(f"✅ Dataset loaded - Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}")
        print(f"🎯 Training for {EPOCHS} epochs...")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        
        # Train
        train_loss, train_acc = train_epoch(
            ddp_model, device, train_loader, optimizer, criterion, epoch, rank
        )
        
        # Test (only compute metrics on rank 0 to avoid duplication)
        test_loss, test_acc = test_model(ddp_model, device, test_loader, rank)
        
        if rank == 0:
            print(f"Epoch {epoch} Results:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\n🎉 Distributed training completed in {total_time:.2f} seconds")
        print(f"Final Test Accuracy: {test_acc:.2f}%")
        
        # Save model (only from rank 0)
        model_path = 'mnist_distributed_tpu_model.pth'
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")


def main():
    """Main function to launch distributed training"""
    print("🚀 Starting Distributed MNIST Training on TPU with PyTorch/XLA")
    
    # Launch multi-process training
    # torch_xla.launch will automatically detect the number of available TPU cores
    torch_xla.launch(_mp_fn)


if __name__ == "__main__":
    main()