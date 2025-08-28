#!/usr/bin/env python3
"""
Simple Transformer Training on Google TPU with PyTorch/XLA
Demonstrates training a small transformer model on synthetic data using TPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.debug.metrics as met
import math
import time
import random
from typing import Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """Simple transformer model for sequence-to-sequence tasks"""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 256, nhead: int = 8, 
                 num_layers: int = 4, dim_feedforward: int = 512, max_seq_len: int = 128):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # src: (seq_len, batch_size)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        if src_mask is None:
            src_mask = self._generate_square_subsequent_mask(src.size(0)).to(src.device)
        
        output = self.transformer(src, src_mask)
        output = self.output_projection(output)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate mask for causal attention"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class SyntheticDataset(Dataset):
    """Generate synthetic sequence data for training"""
    
    def __init__(self, vocab_size: int = 1000, seq_len: int = 64, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        
        # Pre-generate all data for consistency
        self.data = []
        for _ in range(num_samples):
            # Create input sequence
            input_seq = torch.randint(1, vocab_size, (seq_len,))
            
            # Create target sequence (shifted input for next-token prediction)
            target_seq = torch.cat([input_seq[1:], torch.randint(1, vocab_size, (1,))])
            
            self.data.append((input_seq, target_seq))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def create_data_loaders(vocab_size: int = 1000, seq_len: int = 64, batch_size: int = 32):
    """Create training and validation data loaders"""
    train_dataset = SyntheticDataset(vocab_size, seq_len, num_samples=8000)
    val_dataset = SyntheticDataset(vocab_size, seq_len, num_samples=2000)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch"""
    model.train()
    
    # Wrap data loader for XLA
    para_loader = pl.ParallelLoader(train_loader, [device])
    train_loader = para_loader.per_device_loader(device)
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    for batch_idx, (input_seq, target_seq) in enumerate(train_loader):
        with torch_xla.step():
            # input_seq, target_seq: (batch_size, seq_len)
            # Transpose for transformer: (seq_len, batch_size)
            input_seq = input_seq.transpose(0, 1)
            target_seq = target_seq.transpose(0, 1)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(input_seq)  # (seq_len, batch_size, vocab_size)
            
            # Reshape for loss calculation
            output = output.view(-1, output.size(-1))  # (seq_len * batch_size, vocab_size)
            target_seq = target_seq.view(-1)  # (seq_len * batch_size,)
            
            # Calculate loss
            loss = criterion(output, target_seq)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_tokens += target_seq.numel()
            num_batches += 1
        
        # Print progress
        if batch_idx % 50 == 0:
            xm.mark_step()
            avg_loss = total_loss / num_batches
            perplexity = math.exp(min(avg_loss, 10))  # Clip to prevent overflow
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}')
    
    # Final synchronization
    xm.mark_step()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(min(avg_loss, 10))
    
    return avg_loss, perplexity


def validate_model(model, device, val_loader):
    """Validate model performance"""
    model.eval()
    
    # Wrap data loader for XLA
    para_loader = pl.ParallelLoader(val_loader, [device])
    val_loader = para_loader.per_device_loader(device)
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            with torch_xla.step():
                # Transpose for transformer
                input_seq = input_seq.transpose(0, 1)
                target_seq = target_seq.transpose(0, 1)
                
                # Forward pass
                output = model(input_seq)
                
                # Reshape for loss calculation
                output = output.view(-1, output.size(-1))
                target_seq = target_seq.view(-1)
                
                # Calculate loss
                loss = nn.functional.cross_entropy(output, target_seq)
                
                total_loss += loss.item()
                total_tokens += target_seq.numel()
                num_batches += 1
    
    # Final synchronization
    xm.mark_step()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(min(avg_loss, 10))
    
    return avg_loss, perplexity


def main():
    """Main training function"""
    print("🚀 Starting Transformer Training on TPU with PyTorch/XLA")
    
    # Configuration
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    VOCAB_SIZE = 1000
    SEQ_LEN = 64
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 4
    
    # Get XLA device (TPU)
    device = xm.xla_device()
    print(f"Using device: {device}")
    print(f"Device type: {xm.xla_device_hw(device)}")
    
    # Create model and move to TPU
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=N_HEADS,
        num_layers=N_LAYERS,
        max_seq_len=SEQ_LEN
    ).to(device)
    
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # Data loaders
    print("📥 Creating synthetic dataset...")
    train_loader, val_loader = create_data_loaders(
        vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, batch_size=BATCH_SIZE
    )
    print(f"✅ Dataset created - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Training loop
    print(f"\n🎯 Training transformer for {EPOCHS} epochs...")
    start_time = time.time()
    
    best_val_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_loss, train_perplexity = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )
        
        # Validate
        val_loss, val_perplexity = validate_model(model, device, val_loader)
        
        # Update learning rate
        scheduler.step()
        
        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.2f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.2f}")
        print(f"  XLA Metrics: {met.short_metrics_report()}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = 'transformer_tpu_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_path)
            print(f"💾 New best model saved to {model_path}")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {math.exp(min(best_val_loss, 10)):.2f}")


if __name__ == "__main__":
    main()