FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install Python and create virtual environment
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv && \
    python3 -m venv /workspace/venv && \
    . /workspace/venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Set environment variable to use the virtual environment
ENV PATH="/workspace/venv/bin:$PATH"

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python3", "main.py"]
