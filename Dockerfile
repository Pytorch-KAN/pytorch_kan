FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04

# Set working directory
WORKDIR /workspace

# Install Python and essential system packages
RUN apt-get update && apt-get install -y \
    python3-full \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="${PATH}:/root/.local/bin"

# Configure Poetry
RUN poetry config virtualenvs.create false

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock* /workspace/

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the application
COPY . /workspace/

# Set environment variables
ENV PYTHONPATH="/workspace:/workspace/src"

# Set default shell to bash
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/bin/bash"]
