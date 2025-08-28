# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Programming Principles

When writing code for this project, always follow these core principles:

1. **Keep It Simple Stupid (KISS)** - Prefer straightforward, readable solutions over clever or complex ones. Avoid overengineering.
2. **Focused and Concise** - Write targeted code that does one thing well. Avoid feature bloat and unnecessary abstraction.
3. **Break Early and Loudly** - Fail fast with clear, descriptive error messages rather than silent failures or cryptic errors.
4. **Test-Driven Mindset** - Write tests that validate correctness, especially for financial calculations and core business logic.
5. **Idempotency** - Operations should produce the same result when run multiple times, critical for reliable systems.
6. **Explicit is Better Than Implicit** - Make dependencies, assumptions, and data flows obvious in the code.

These principles should guide all implementation decisions, from function design to error handling to API interfaces.

## Overview

This repository contains PyTorch TPU training examples using PyTorch/XLA 2.8.0 (2025). It includes comprehensive examples for training neural networks on Google Cloud TPU, from simple MNIST classification to distributed transformer training.

## Key Commands

### Environment Setup
```bash
# Verify TPU environment and dependencies
python setup_tpu.py

# Install all dependencies
pip install -r requirements.txt
```

### Training Examples
```bash
# Simple MNIST training on single TPU
python simple_mnist_tpu.py

# Distributed MNIST training on multiple TPUs
python distributed_mnist_tpu.py

# Transformer training with synthetic data
python transformer_tpu.py

# Load and test different configurations
python config.py
```

### Docker Deployment
```bash
# Build TPU environment container
docker build -t pytorch-tpu .

# Run container with TPU support
docker run -it --rm pytorch-tpu
```

### Google Cloud TPU Commands
```bash
# Create TPU VM (v5litepod-8 recommended for 2025)
gcloud compute tpus tpu-vm create tpu-training \
  --zone=us-central2-b \
  --accelerator-type=v5litepod-8 \
  --version=tpu-ubuntu2204-base

# Connect to TPU VM
gcloud compute tpus tpu-vm ssh tpu-training --zone=us-central2-b

# Delete TPU VM when done
gcloud compute tpus tpu-vm delete tpu-training --zone=us-central2-b
```

## Architecture

### Core Components

- **Training Scripts**: Three main examples demonstrating different TPU usage patterns
  - `simple_mnist_tpu.py`: Single TPU training with basic CNN
  - `distributed_mnist_tpu.py`: Multi-TPU distributed training using DDP
  - `transformer_tpu.py`: Advanced transformer training with synthetic data

- **Utilities Layer**: `tpu_utils.py` provides essential TPU operations
  - Device management and information
  - Performance monitoring and profiling
  - Memory management utilities
  - Checkpoint save/load with TPU considerations

- **Configuration System**: `config.py` centralizes all training parameters
  - Predefined configuration presets (quick_test, development, production, distributed)
  - TPU-specific optimizations (BF16, XLA settings)
  - Optimizer and scheduler factory functions

- **Environment Management**: `setup_tpu.py` validates the complete TPU stack
  - Python 3.11+ compatibility check
  - PyTorch/XLA 2.8.0 installation verification
  - TPU device detection and basic operation testing

### TPU-Specific Patterns

All training scripts follow 2025 PyTorch/XLA best practices:
- Use `torch_xla.step()` context manager for optimal XLA graph compilation
- Implement proper device placement with `model.to('xla')` and `data.to('xla')`
- Use `ParallelLoader` for efficient data pipeline
- Include `xm.mark_step()` for synchronization at appropriate points
- Enable mixed precision (BF16) for better performance

### Data Flow Architecture

1. **Data Loading**: Synthetic datasets (MNIST, sequence data) with distributed sampling
2. **Model Execution**: XLA compilation and execution on TPU cores
3. **Gradient Synchronization**: Automatic via DistributedDataParallel for multi-TPU
4. **Monitoring**: XLA metrics collection and performance reporting

## Dependencies

### Core Dependencies (PyTorch/XLA 2.8.0 - 2025)
- `torch==2.8.0`: Core PyTorch framework
- `torch_xla[tpu]==2.8.0`: PyTorch/XLA TPU support with Python 3.11+ compatibility
- `torchvision==0.19.0`: Vision utilities for MNIST example

### ML and Data Processing
- `numpy>=1.21.0`: Numerical computing
- `scikit-learn>=1.1.0`: ML utilities
- `matplotlib>=3.5.0`: Plotting and visualization

### Monitoring and Profiling  
- `tensorboard>=2.10.0`: Training visualization
- `tensorboard-plugin-profile>=2.13.0`: XLA profiling plugin
- `tqdm>=4.64.0`: Progress bars

### Cloud Integration
- `google-cloud-storage>=2.10.0`: GCS integration for model/data storage

### Development Tools
- `pytest>=7.0.0`: Testing framework
- `black>=22.0.0`: Code formatting
- `flake8>=5.0.0`: Linting

### Environment Variables
Key TPU environment variables automatically configured:
- `PJRT_DEVICE=TPU`: Use PJRT runtime for TPU
- `XLA_USE_BF16=1`: Enable mixed precision training
- `TPU_NUM_DEVICES=8`: Default for v5litepod-8

## Important Notes

- All scripts should use proper error handling and validation
- Prefer explicit configuration over implicit defaults
- Document any non-obvious business logic or calculations
- Use descriptive variable and function names
- Write tests for critical functionality

## Important Instruction Reminders

Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.