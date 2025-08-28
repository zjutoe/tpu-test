#!/usr/bin/env python3
import os
os.environ['PJRT_DEVICE'] = 'TPU'
os.environ['TPU_NUM_DEVICES'] = '8'

import torch
import torch_xla
import torch_xla.core.xla_model as xm

print("PyTorch version:", torch.__version__)
print("PyTorch/XLA version:", torch_xla.__version__)

try:
    # Get TPU device
    device = xm.xla_device()
    print("✅ TPU device:", device)
    
    # Test basic operations
    x = torch.randn(3, 3, device=device)
    y = x + 1
    xm.mark_step()
    print("✅ Basic TPU operations successful")
    print("Tensor shape:", y.shape)
    
except Exception as e:
    print("❌ TPU test failed:", str(e))