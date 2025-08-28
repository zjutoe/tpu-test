# Dockerfile for PyTorch TPU Training Environment
# Based on Google Cloud TPU VM with PyTorch/XLA 2.8.0 (2025)

FROM gcr.io/deeplearning-platform-release/pytorch-tpu:2.8.0-py311

# Set working directory
WORKDIR /workspace

# Set environment variables for TPU
ENV PJRT_DEVICE=TPU
ENV TPU_NUM_DEVICES=8
ENV XLA_USE_BF16=1
ENV PYTHONPATH=/workspace:$PYTHONPATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    htop \
    tmux \
    screen \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Copy requirements and install Python packages
COPY requirements.txt /workspace/
RUN pip install -r requirements.txt

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/data \
    /workspace/checkpoints \
    /workspace/logs \
    /workspace/models

# Set proper permissions
RUN chmod +x /workspace/*.py

# Create a non-root user for better security
RUN useradd -m -u 1000 tpuuser && \
    chown -R tpuuser:tpuuser /workspace
USER tpuuser

# Verify PyTorch/XLA installation
RUN python -c "import torch; import torch_xla; print('PyTorch:', torch.__version__); print('PyTorch/XLA:', torch_xla.__version__)"

# Set default command
CMD ["bash"]

# Labels
LABEL maintainer="TPU Training Environment"
LABEL description="PyTorch/XLA 2.8.0 environment for Google Cloud TPU training"
LABEL version="2025.1"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch_xla.core.xla_model as xm; print('TPU Health:', xm.get_xla_supported_devices())" || exit 1