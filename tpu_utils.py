#!/usr/bin/env python3
"""
TPU Utilities for PyTorch/XLA Training
Collection of helper functions for TPU operations, monitoring, and optimization.
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met
import torch_xla.distributed.parallel_loader as pl
import functools
import time
import os
from typing import Dict, List, Any, Optional, Callable
import logging


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_tpu_device() -> torch.device:
    """Setup and return TPU device with basic configuration"""
    device = xm.xla_device()
    logger.info(f"TPU device: {device}")
    logger.info(f"Device type: {xm.xla_device_hw(device)}")
    logger.info(f"World size: {xm.xrt_world_size()}")
    logger.info(f"Ordinal: {xm.get_ordinal()}")
    return device


def get_tpu_info() -> Dict[str, Any]:
    """Get comprehensive TPU information"""
    try:
        device = xm.xla_device()
        info = {
            'device': str(device),
            'device_type': xm.xla_device_hw(device),
            'world_size': xm.xrt_world_size(),
            'ordinal': xm.get_ordinal(),
            'is_master_ordinal': xm.is_master_ordinal(),
            'supported_devices': xm.get_xla_supported_devices(),
        }
        
        # Add environment variables
        env_vars = ['TPU_NAME', 'TPU_ZONE', 'PJRT_DEVICE', 'XLA_USE_BF16']
        for var in env_vars:
            info[f'env_{var.lower()}'] = os.environ.get(var, 'Not set')
            
        return info
    except Exception as e:
        logger.error(f"Error getting TPU info: {e}")
        return {}


def print_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Print and return model size information"""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (approximate)
    param_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32
    
    info = {
        'total_parameters': param_count,
        'trainable_parameters': trainable_count,
        'estimated_size_mb': param_size_mb
    }
    
    print(f"Model Parameters:")
    print(f"  Total: {param_count:,}")
    print(f"  Trainable: {trainable_count:,}")
    print(f"  Estimated size: {param_size_mb:.2f} MB")
    
    return info


class TPUTimer:
    """Context manager for timing TPU operations with proper synchronization"""
    
    def __init__(self, name: str = "Operation", sync: bool = True):
        self.name = name
        self.sync = sync
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        if self.sync:
            xm.mark_step()  # Ensure previous operations are complete
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if self.sync:
            xm.mark_step()  # Ensure current operations are complete
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"{self.name} took {duration:.4f} seconds")
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


def safe_barrier():
    """Safely synchronize all TPU processes"""
    try:
        xm.rendezvous('safe_barrier')
        logger.info("TPU barrier completed successfully")
    except Exception as e:
        logger.warning(f"TPU barrier failed: {e}")


def optimize_for_tpu(func: Callable) -> Callable:
    """Decorator to optimize functions for TPU execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch_xla.step():
            result = func(*args, **kwargs)
        return result
    return wrapper


class TPUMetricsTracker:
    """Track and report TPU metrics during training"""
    
    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
        self.metrics_history = []
    
    def step(self):
        """Record metrics for current step"""
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            metrics = met.metrics_report()
            self.metrics_history.append({
                'step': self.step_count,
                'metrics': metrics,
                'short_report': met.short_metrics_report()
            })
            
            logger.info(f"Step {self.step_count} - {met.short_metrics_report()}")
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest recorded metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return {}
    
    def clear_metrics(self):
        """Clear XLA metrics"""
        met.clear_metrics()
        logger.info("XLA metrics cleared")


def create_parallel_loader(dataloader, devices: List[torch.device] = None):
    """Create a parallel data loader for TPU training"""
    if devices is None:
        devices = [xm.xla_device()]
    
    para_loader = pl.ParallelLoader(dataloader, devices)
    return para_loader


def move_to_device(data, device: torch.device):
    """Recursively move data structures to TPU device"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(move_to_device(item, device) for item in data)
    else:
        return data


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str, **kwargs):
    """Save model checkpoint with TPU-specific considerations"""
    # Only save from master ordinal to avoid conflicts
    if xm.is_master_ordinal():
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'tpu_info': get_tpu_info(),
            **kwargs
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    # Barrier to ensure all processes wait for save to complete
    safe_barrier()


def load_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                   filepath: str) -> Dict[str, Any]:
    """Load model checkpoint on TPU"""
    device = xm.xla_device()
    
    # Load checkpoint
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model and optimizer state
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {filepath}")
    logger.info(f"Resumed from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


def profile_step(step_fn: Callable, *args, **kwargs):
    """Profile a single training step on TPU"""
    with TPUTimer("Training step", sync=True):
        result = step_fn(*args, **kwargs)
    
    # Print XLA compilation metrics
    print("XLA Compilation metrics:", met.short_metrics_report())
    
    return result


def get_memory_info() -> Dict[str, Any]:
    """Get TPU memory information (best effort)"""
    try:
        device = xm.xla_device()
        
        # This is approximate - TPU memory management is handled by XLA
        info = {
            'device': str(device),
            'xla_metrics': met.short_metrics_report(),
        }
        
        return info
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return {}


def configure_mixed_precision(use_bf16: bool = True):
    """Configure mixed precision training for TPU"""
    if use_bf16:
        os.environ['XLA_USE_BF16'] = '1'
        logger.info("Mixed precision (BF16) enabled")
    else:
        os.environ.pop('XLA_USE_BF16', None)
        logger.info("Mixed precision (BF16) disabled")


class TPUProfiler:
    """Simple TPU profiling context manager"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.start_metrics = None
        
    def __enter__(self):
        if self.enabled:
            met.clear_metrics()
            self.start_metrics = met.metrics_report()
        return self
        
    def __exit__(self, *args):
        if self.enabled:
            end_metrics = met.metrics_report()
            print("=== TPU Profiling Results ===")
            print(met.metrics_report())
            print("=============================")


# Convenience functions
def is_tpu_available() -> bool:
    """Check if TPU is available"""
    try:
        device = xm.xla_device()
        return 'TPU' in str(device)
    except:
        return False


def get_world_size() -> int:
    """Get TPU world size"""
    try:
        return xm.xrt_world_size()
    except:
        return 1


def get_rank() -> int:
    """Get current TPU rank/ordinal"""
    try:
        return xm.get_ordinal()
    except:
        return 0


def is_master() -> bool:
    """Check if current process is master"""
    try:
        return xm.is_master_ordinal()
    except:
        return True