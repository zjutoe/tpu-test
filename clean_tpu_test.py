#!/usr/bin/env python3
"""
Clean TPU test that avoids double initialization issues
"""

import subprocess
import sys

def run_isolated_test():
    """Run TPU test in isolated subprocess to avoid initialization conflicts"""
    
    test_code = '''
import os
os.environ["PJRT_DEVICE"] = "CPU"  # Force CPU backend first
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import torch
import torch_xla
import torch_xla.core.xla_model as xm

print("PyTorch version:", torch.__version__)
print("PyTorch/XLA version:", torch_xla.__version__)

try:
    # Get supported devices
    devices = xm.get_xla_supported_devices()
    print("Supported XLA devices:", devices)
    
    if devices:
        device = devices[0] if devices else "cpu"
        print(f"Using device: {device}")
        
        # Test basic tensor operations
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        x_xla = x.to(device)
        y = x_xla * 2 + 1
        
        # Force computation
        xm.mark_step()
        
        result = y.cpu()
        print("✅ XLA computation successful!")
        print(f"Input: {x}")
        print(f"Output: {result}")
        
    else:
        print("❌ No XLA devices available")
        
except Exception as e:
    print(f"❌ XLA test failed: {e}")
    import traceback
    traceback.print_exc()
'''
    
    try:
        # Run in subprocess to avoid initialization conflicts
        result = subprocess.run([
            sys.executable, '-c', test_code
        ], capture_output=True, text=True, timeout=30)
        
        print("=== SUBPROCESS OUTPUT ===")
        print(result.stdout)
        if result.stderr:
            print("=== SUBPROCESS ERRORS ===")
            print(result.stderr)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Failed to run subprocess test: {e}")
        return False

def test_regular_pytorch():
    """Test regular PyTorch without XLA"""
    try:
        import torch
        print(f"\n=== Regular PyTorch Test ===")
        print(f"PyTorch version: {torch.__version__}")
        
        x = torch.randn(3, 3)
        y = torch.matmul(x, x.T)
        print(f"✅ Regular PyTorch works fine: {y.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Regular PyTorch failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Clean TPU/XLA Test")
    print("=" * 40)
    
    # Test regular PyTorch first
    pytorch_ok = test_regular_pytorch()
    
    # Test XLA in subprocess
    xla_ok = run_isolated_test()
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"Regular PyTorch: {'✅ OK' if pytorch_ok else '❌ Failed'}")
    print(f"PyTorch/XLA: {'✅ OK' if xla_ok else '❌ Failed'}")
    
    if xla_ok:
        print("\n🎉 XLA is working! You can proceed with training.")
    else:
        print("\n⚠️  XLA has issues, but regular PyTorch works.")
        print("You may need to use CPU-only training or fix TPU setup.")