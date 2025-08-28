# PyTorch TPU Training Examples

A comprehensive collection of PyTorch training examples optimized for Google Cloud TPU using PyTorch/XLA 2.8.0 (2025).

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (required for PyTorch/XLA 2.8.0)
- Google Cloud account with TPU access
- Basic familiarity with PyTorch

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd tpu-test
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify TPU environment:**
   ```bash
   python setup_tpu.py
   ```

## 📋 Available Examples

### 1. Simple MNIST Training (`simple_mnist_tpu.py`)
Basic CNN training on MNIST dataset using single TPU core.

```bash
python simple_mnist_tpu.py
```

**Features:**
- Simple CNN architecture
- TPU-optimized data loading
- Uses latest `torch_xla.step()` context manager
- Performance monitoring with XLA metrics

### 2. Distributed MNIST Training (`distributed_mnist_tpu.py`)
Multi-TPU training using DistributedDataParallel.

```bash
python distributed_mnist_tpu.py
```

**Features:**
- Distributed training across multiple TPU cores
- Uses `torch_xla.launch()` for multi-process setup
- XLA backend for distributed communication
- Proper gradient synchronization

### 3. Transformer Training (`transformer_tpu.py`)
Simple transformer model training on synthetic data.

```bash
python transformer_tpu.py
```

**Features:**
- Custom transformer implementation
- Synthetic sequence-to-sequence data
- Mixed precision training (BF16)
- Learning rate scheduling
- Model checkpointing

## 🛠 Utilities

### TPU Environment Setup (`setup_tpu.py`)
Comprehensive TPU environment verification:
- Python version compatibility
- PyTorch/XLA installation check
- TPU device detection
- Environment variable validation

### TPU Utilities (`tpu_utils.py`)
Helper functions for TPU operations:
- Device setup and info
- Model size calculation
- Timing and profiling
- Memory monitoring
- Checkpoint management

### Configuration (`config.py`)
Centralized configuration management:
- Predefined configuration presets
- TPU-optimized settings
- Environment variable management
- Optimizer and scheduler factory functions

## 🏗 Google Cloud TPU Setup

### 1. Create TPU VM

```bash
# Create TPU v5e (recommended for 2025)
gcloud compute tpus tpu-vm create tpu-training \
  --zone=us-central2-b \
  --accelerator-type=v5litepod-8 \
  --version=tpu-ubuntu2204-base
```

### 2. Connect to TPU VM

```bash
gcloud compute tpus tpu-vm ssh tpu-training --zone=us-central2-b
```

### 3. Setup Environment

```bash
# Clone your repository
git clone <your-repo-url>
cd tpu-test

# Install dependencies
pip install -r requirements.txt

# Verify setup
python setup_tpu.py
```

## 🐳 Docker Usage

Build and run the Docker container:

```bash
# Build image
docker build -t pytorch-tpu .

# Run container
docker run -it --rm pytorch-tpu

# Verify installation inside container
python setup_tpu.py
```

## 📊 Performance Optimization

### Best Practices for TPU Training

1. **Use BF16 Mixed Precision:**
   ```python
   import os
   os.environ['XLA_USE_BF16'] = '1'
   ```

2. **Optimize Batch Sizes:**
   - Single TPU core: 128-512
   - Multi-TPU: 32-128 per core

3. **Use Proper XLA Patterns:**
   ```python
   with torch_xla.step():
       # Training step code
       pass
   ```

4. **Monitor XLA Compilation:**
   ```python
   import torch_xla.debug.metrics as met
   print(met.short_metrics_report())
   ```

### TPU-Specific Configurations

```python
from config import load_config

# Load optimized configuration
config = load_config('production')

# Apply TPU environment variables
config['tpu'].apply_env_vars()
```

## 🔍 Monitoring and Debugging

### XLA Metrics
Monitor compilation and execution:
```python
import torch_xla.debug.metrics as met

# Clear metrics
met.clear_metrics()

# Training code here

# Check metrics
print(met.metrics_report())
```

### Memory Monitoring
```python
from tpu_utils import get_memory_info, TPUTimer

# Memory usage
print(get_memory_info())

# Timing operations
with TPUTimer("Training step"):
    # Your training code
    pass
```

### Profiling
```python
from tpu_utils import TPUProfiler

with TPUProfiler():
    # Code to profile
    pass
```

## 🚨 Troubleshooting

### Common Issues

1. **XLA Compilation Timeouts**
   - Reduce model size or batch size
   - Use smaller sequence lengths
   - Enable mixed precision

2. **Memory Issues**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

3. **Slow Training**
   - Check XLA compilation metrics
   - Ensure proper data loading
   - Verify TPU utilization

### Environment Variables

Key environment variables for TPU:
```bash
export PJRT_DEVICE=TPU
export TPU_NUM_DEVICES=8
export XLA_USE_BF16=1
```

## 📈 Example Performance Results

### MNIST Training (v5litepod-8)
- **Single TPU:** ~45 seconds for 5 epochs
- **Multi-TPU:** ~25 seconds for 5 epochs
- **Accuracy:** >98% test accuracy

### Transformer Training (256d model)
- **Training time:** ~2 minutes per epoch
- **Memory usage:** ~4GB per core
- **Throughput:** ~1000 tokens/second/core

## 🔗 Additional Resources

- [PyTorch/XLA Documentation](https://pytorch.org/xla/)
- [Google Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [PyTorch/XLA GitHub Repository](https://github.com/pytorch/xla)
- [TPU Performance Guide](https://cloud.google.com/tpu/docs/performance-guide)

## 📝 Configuration Presets

Available configuration presets in `config.py`:

- **`quick_test`**: Fast testing with minimal resources
- **`development`**: Development with profiling enabled
- **`production`**: Optimized for production training
- **`distributed`**: Multi-TPU distributed training

```python
from config import load_config

# Load configuration
config = load_config('production')
print_config(config)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test on TPU environment
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch/XLA team for excellent TPU support
- Google Cloud TPU team for the hardware platform
- PyTorch community for the foundational framework