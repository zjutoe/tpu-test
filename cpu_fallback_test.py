#!/usr/bin/env python3
"""
Test PyTorch/XLA with CPU fallback when TPU is not available
"""

import os
import torch
import torch_xla

print("PyTorch version:", torch.__version__)
print("PyTorch/XLA version:", torch_xla.__version__)

# Try different device configurations
test_configs = [
    {"PJRT_DEVICE": "TPU", "TPU_NUM_DEVICES": "8"},
    {"PJRT_DEVICE": "CPU", "XLA_FLAGS": "--xla_force_host_platform_device_count=8"},
    {}  # No environment variables
]

for i, config in enumerate(test_configs):
    print(f"\n--- Test Configuration {i+1} ---")
    
    # Set environment variables
    for key, value in config.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    try:
        import torch_xla.core.xla_model as xm
        
        # Get supported devices
        supported_devices = xm.get_xla_supported_devices()
        print(f"Supported devices: {supported_devices}")
        
        if supported_devices:
            # Try to use the first available device
            device = supported_devices[0]
            print(f"Using device: {device}")
            
            # Test basic operations
            x = torch.randn(3, 3)
            x_xla = x.to(device)
            y = x_xla + 1
            
            # Force execution
            xm.mark_step()
            
            # Get result back to CPU
            result = y.cpu()
            print(f"✅ Success! Result shape: {result.shape}")
            print(f"Sample values: {result[0, 0].item():.4f}")
            
            break  # Exit loop on first success
            
        else:
            print("❌ No XLA devices found")
            
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        
    # Clean up environment variables for next test
    for key in config.keys():
        os.environ.pop(key, None)

print(f"\n--- Final Test: Simple PyTorch (no XLA) ---")
try:
    x = torch.randn(3, 3)
    y = x + 1
    print(f"✅ Regular PyTorch works: {y.shape}")
except Exception as e:
    print(f"❌ Even regular PyTorch failed: {e}")