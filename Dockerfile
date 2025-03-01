FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04

# Set working directory
WORKDIR /workspace

# Create src directory
RUN mkdir -p /workspace/src

# Copy requirements file
COPY requirements.txt /workspace/

# Install Python and create virtual environment
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && \
    python3 -m venv /workspace/venv && \
    . /workspace/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Set environment variable to use the virtual environment
ENV PATH="/workspace/venv/bin:$PATH"
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"

# We don't copy any code here - it will be mounted from host

# Command to run the application (adjust as needed)
CMD ["python3", "/workspace/src/main.py"]
