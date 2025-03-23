FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04

# Set working directory
WORKDIR /workspace

# Create src directory
RUN mkdir -p /workspace/src

# Copy requirements file
COPY requirements.txt /workspace/

# Install Python and essential system packages only
RUN apt-get update && apt-get install -y \
    python3-full \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH="/workspace:/workspace/src"

# Set default shell to bash
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash"]
