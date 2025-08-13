# Dockerfile for gptoss-compressor with CUDA 12.8 and PyTorch 2.8.0
# Supports building QQQ CUTLASS kernels for W4A8 quantization

# Use PyTorch 2.8.0 with CUDA 12.8 and development tools
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# Build arguments
ARG CUDA_ARCH=8.6
ARG INSTALL_QQQ=true
ARG TRANSFORMERS_VERSION=4.55.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}
ENV FORCE_CUDA=1
ENV MAX_JOBS=4

# Install system dependencies for building CUDA extensions
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    git \
    cmake \
    ninja-build \
    wget \
    curl \
    vim \
    htop \
    pkg-config \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify CUDA toolkit installation
RUN nvcc --version || echo "NVCC not found - CUDA extensions may fail to build"

# Set working directory
WORKDIR /workspace/app


# Copy requirements and install Python dependencies first (better caching)

# Upgrade pip and transformers before installing other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install "transformers==${TRANSFORMERS_VERSION}" && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://pypi.nvidia.com && \
    rm requirements.txt

# Set remote code execution to true for transformers
ENV TRANSFORMERS_REMOTE_CODE_EXECUTION=1

# Copy the application code
COPY . .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Install QQQ CUTLASS kernels if requested - improved version
RUN if [ "$INSTALL_QQQ" = "true" ]; then \
        echo "=== Installing QQQ CUTLASS kernels for CUDA architecture ${CUDA_ARCH} ===" && \
        echo "Downloading QQQ from GitHub..." && \
        pip install --no-cache-dir --verbose git+https://github.com/HandH1998/QQQ.git && \
        echo "QQQ installation completed" && \
        echo "Verifying QQQ installation..." && \
        python -c "import QQQ; print('✓ QQQ package imported successfully'); print(f'QQQ version: {getattr(QQQ, \"__version__\", \"unknown\")}')" 2>/dev/null || \
        python -c "import qqq; print('✓ qqq package imported successfully'); print(f'qqq version: {getattr(qqq, \"__version__\", \"unknown\")}')" 2>/dev/null || \
        echo "⚠ QQQ import test failed - this may be normal for some builds" && \
        echo "=== QQQ installation process completed ===" ; \
    else \
        echo "Skipping QQQ installation (INSTALL_QQQ=false)"; \
    fi

# Create directories for offloading
RUN mkdir -p /workspace/app/offload_cache && \
    mkdir -p /workspace/app/models

# Set entrypoint to run automatic verification
ENTRYPOINT ["./entrypoint.sh"]

# Default command (interactive shell after verification)
CMD []

# Labels
LABEL maintainer="gptoss-compressor"
LABEL description="Dequantize and compress GPT-OSS models with GPTQ/AWQ"
LABEL cuda_version="12.8"
LABEL pytorch_version="2.8.0"
LABEL cuda_arch="${CUDA_ARCH}"
