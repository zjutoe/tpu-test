#!/usr/bin/env python3
"""
TPU Environment Setup and Verification Script
Verifies TPU availability and environment configuration for PyTorch/XLA
"""

import sys
import os
from typing import List, Dict, Any


def check_python_version():
    """Check if Python version is compatible with PyTorch/XLA 2.8.0"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 11:
        print("❌ Error: Python 3.11+ is required for PyTorch/XLA 2.8.0")
        return False
    
    print("✅ Python version is compatible")
    return True


def check_torch_xla():
    """Check if PyTorch and XLA are properly installed"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        import torch_xla
        print(f"✅ PyTorch/XLA version: {torch_xla.__version__}")
        
        import torch_xla.core.xla_model as xm
        print("✅ PyTorch/XLA core modules imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Error importing PyTorch/XLA: {e}")
        print("Run: pip install torch==2.8.0 'torch_xla[tpu]==2.8.0'")
        return False


def check_tpu_devices():
    """Check available TPU devices"""
    try:
        import torch_xla.core.xla_model as xm
        
        devices = xm.get_xla_supported_devices()
        print(f"✅ XLA supported devices: {devices}")
        
        tpu_devices = [d for d in devices if 'TPU' in d]
        if tpu_devices:
            print(f"✅ TPU devices found: {tpu_devices}")
            
            # Try to get device properties
            device = xm.xla_device()
            print(f"✅ Current XLA device: {device}")
            
            # Test basic tensor operations
            x = torch.randn(3, 3).to(device)
            y = x + 1
            xm.mark_step()  # Force execution
            print("✅ Basic tensor operations on TPU successful")
            
            return True
        else:
            print("⚠️  No TPU devices found. Running on CPU/GPU instead.")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing TPU devices: {e}")
        return False


def check_environment_variables():
    """Check important environment variables for TPU"""
    important_vars = [
        'TPU_NAME',
        'TPU_ZONE', 
        'PJRT_DEVICE',
        'XLA_USE_BF16',
        'TPU_NUM_DEVICES'
    ]
    
    print("\nEnvironment variables:")
    for var in important_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")


def get_device_info():
    """Get detailed device information"""
    try:
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        print(f"\nDevice Information:")
        print(f"  Device: {device}")
        print(f"  Device type: {xm.xla_device_hw(device)}")
        print(f"  World size: {xm.xrt_world_size()}")
        print(f"  Ordinal: {xm.get_ordinal()}")
        
    except Exception as e:
        print(f"Could not get device info: {e}")


def main():
    """Main setup verification function"""
    print("🚀 PyTorch TPU Environment Verification\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch/XLA Installation", check_torch_xla),
        ("TPU Device Access", check_tpu_devices),
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n--- {check_name} ---")
        result = check_func()
        results.append((check_name, result))
    
    check_environment_variables()
    get_device_info()
    
    print(f"\n{'='*50}")
    print("VERIFICATION SUMMARY:")
    print(f"{'='*50}")
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print(f"\n🎉 All checks passed! TPU environment is ready.")
        print("You can now run the training scripts.")
    else:
        print(f"\n⚠️  Some checks failed. Please resolve issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()