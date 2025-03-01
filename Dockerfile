FROM nvcr.io/nvidia/cuda-dl-base:24.12-cuda12.6-devel-ubuntu24.04

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install python3-pip -y && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python3", "main.py"]
