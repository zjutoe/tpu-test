#!/usr/bin/env python3
"""
Configuration file for TPU training experiments
Centralized configuration for all training scripts with TPU-optimized settings.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class TPUConfig:
    """TPU-specific configuration"""
    # Device settings
    use_bf16: bool = True
    compile_model: bool = True
    
    # Performance settings
    prefetch_factor: int = 2
    num_workers: int = 4
    pin_memory: bool = True
    
    # XLA settings
    xla_sync_wait: bool = False
    mark_step_frequency: int = 1
    
    # Environment variables to set
    env_vars: Dict[str, str] = None
    
    def __post_init__(self):
        if self.env_vars is None:
            self.env_vars = {
                'XLA_USE_BF16': '1' if self.use_bf16 else '0',
                'TPU_NUM_DEVICES': '8',  # Default for v5litepod-8
                'PJRT_DEVICE': 'TPU',
            }
    
    def apply_env_vars(self):
        """Apply environment variables"""
        for key, value in self.env_vars.items():
            os.environ[key] = value
            print(f"Set {key}={value}")


@dataclass 
class MNISTConfig:
    """Configuration for MNIST training"""
    # Data settings
    batch_size: int = 128
    num_workers: int = 4
    data_dir: str = './data'
    
    # Model settings
    num_classes: int = 10
    dropout_rate: float = 0.25
    
    # Training settings
    epochs: int = 5
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    
    # Optimizer settings
    optimizer: str = 'adam'  # 'adam', 'sgd', 'adamw'
    momentum: float = 0.9
    
    # Scheduler settings
    use_scheduler: bool = True
    scheduler_type: str = 'step'  # 'step', 'cosine'
    step_size: int = 2
    gamma: float = 0.1
    
    # Logging
    log_interval: int = 100
    save_model: bool = True
    model_save_path: str = 'mnist_model.pth'


@dataclass
class DistributedMNISTConfig(MNISTConfig):
    """Configuration for distributed MNIST training"""
    batch_size: int = 32  # Per-device batch size
    epochs: int = 3  # Fewer epochs for distributed example


@dataclass
class TransformerConfig:
    """Configuration for transformer training"""
    # Model architecture
    vocab_size: int = 1000
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 512
    max_seq_len: int = 128
    dropout: float = 0.1
    
    # Data settings
    seq_len: int = 64
    batch_size: int = 32
    num_train_samples: int = 8000
    num_val_samples: int = 2000
    
    # Training settings
    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    
    # Optimizer settings
    optimizer: str = 'adamw'
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # Scheduler settings
    use_scheduler: bool = True
    scheduler_type: str = 'step'
    step_size: int = 3
    gamma: float = 0.1
    
    # Logging and checkpointing
    log_interval: int = 50
    save_best: bool = True
    checkpoint_dir: str = './checkpoints'
    model_save_path: str = 'transformer_model.pth'


@dataclass
class TrainingConfig:
    """General training configuration"""
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_interval: int = 1  # Save every N epochs
    max_checkpoints: int = 5
    
    # Early stopping
    use_early_stopping: bool = False
    patience: int = 3
    min_delta: float = 1e-4
    
    # Validation
    validate_every: int = 1  # Validate every N epochs
    
    # Profiling and monitoring
    profile_training: bool = False
    monitor_memory: bool = True
    log_metrics: bool = True


# Predefined configurations for different scenarios
class ConfigPresets:
    """Predefined configuration presets"""
    
    @staticmethod
    def get_quick_test() -> Dict[str, Any]:
        """Configuration for quick testing"""
        return {
            'tpu': TPUConfig(use_bf16=False),
            'mnist': MNISTConfig(epochs=1, batch_size=64),
            'transformer': TransformerConfig(epochs=2, batch_size=16, num_train_samples=1000),
            'training': TrainingConfig(save_checkpoints=False)
        }
    
    @staticmethod
    def get_development() -> Dict[str, Any]:
        """Configuration for development"""
        return {
            'tpu': TPUConfig(),
            'mnist': MNISTConfig(epochs=3),
            'transformer': TransformerConfig(epochs=5),
            'training': TrainingConfig(profile_training=True)
        }
    
    @staticmethod
    def get_production() -> Dict[str, Any]:
        """Configuration for production training"""
        return {
            'tpu': TPUConfig(use_bf16=True, compile_model=True),
            'mnist': MNISTConfig(),
            'transformer': TransformerConfig(
                epochs=20,
                batch_size=64,
                num_train_samples=50000,
                learning_rate=0.0005
            ),
            'training': TrainingConfig(
                save_checkpoints=True,
                use_early_stopping=True,
                profile_training=False
            )
        }
    
    @staticmethod
    def get_distributed() -> Dict[str, Any]:
        """Configuration for distributed training"""
        return {
            'tpu': TPUConfig(use_bf16=True),
            'mnist': DistributedMNISTConfig(),
            'training': TrainingConfig(save_checkpoints=True)
        }


def load_config(preset: str = 'development') -> Dict[str, Any]:
    """Load configuration preset"""
    presets = {
        'quick_test': ConfigPresets.get_quick_test,
        'development': ConfigPresets.get_development,
        'production': ConfigPresets.get_production,
        'distributed': ConfigPresets.get_distributed
    }
    
    if preset not in presets:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(presets.keys())}")
    
    config = presets[preset]()
    
    # Apply TPU environment variables
    config['tpu'].apply_env_vars()
    
    return config


def get_optimizer(model_parameters, config):
    """Create optimizer based on configuration"""
    if hasattr(config, 'optimizer'):
        opt_name = config.optimizer.lower()
    else:
        opt_name = 'adam'
    
    lr = config.learning_rate
    weight_decay = getattr(config, 'weight_decay', 0.0)
    
    if opt_name == 'adam':
        return torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        betas = getattr(config, 'betas', (0.9, 0.999))
        eps = getattr(config, 'eps', 1e-8)
        return torch.optim.AdamW(
            model_parameters, lr=lr, weight_decay=weight_decay, 
            betas=betas, eps=eps
        )
    elif opt_name == 'sgd':
        momentum = getattr(config, 'momentum', 0.9)
        return torch.optim.SGD(
            model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def get_scheduler(optimizer, config):
    """Create learning rate scheduler based on configuration"""
    if not getattr(config, 'use_scheduler', False):
        return None
    
    scheduler_type = getattr(config, 'scheduler_type', 'step').lower()
    
    if scheduler_type == 'step':
        step_size = getattr(config, 'step_size', 3)
        gamma = getattr(config, 'gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = getattr(config, 'epochs', 10)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def print_config(config: Dict[str, Any]):
    """Print configuration in a readable format"""
    print("=== Training Configuration ===")
    for section_name, section_config in config.items():
        print(f"\n{section_name.upper()}:")
        if hasattr(section_config, '__dict__'):
            for key, value in section_config.__dict__.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {section_config}")
    print("=" * 30)


# Example usage
if __name__ == "__main__":
    # Load and print development configuration
    config = load_config('development')
    print_config(config)