FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04

# Set working directory
WORKDIR /workspace

# Create src directory
RUN mkdir -p /workspace/src

# Copy requirements file
COPY requirements.txt /workspace/

# Install Python and system packages
RUN apt-get update && apt-get install -y \
    python3-full \
    python3-pip \
    python3-dev \
    python3-numpy \
    python3-pandas \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and other ML packages first
RUN pip3 install --break-system-packages torch torchvision torchaudio

# Install additional Python packages
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH="/workspace:/workspace/src"

# Set default shell to bash
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash"]
