# Variables
IMAGE_NAME = pytorch-kan
IMAGE_NAME_CUDA = pytorch-kan:cuda
CONTAINER_NAME = pytorch-kan
WORKSPACE_DIR = $(shell pwd)
VENV_NAME = kan_venv
PYTHON = python3

# Step 1: Create required volumes
setup:
	docker volume create vscode-server
	@echo "Volumes created successfully"

# Step 2: Build the Docker image (use build-no-cache for fresh builds)
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) .

build-no-cache:
	@echo "Building Docker image without cache..."
	docker build --no-cache -t $(IMAGE_NAME) .

build-cuda:
	@echo "Building CUDA Docker image..."
	docker build -f Dockerfile.cuda -t $(IMAGE_NAME_CUDA) .

## Step 3: Run the container
container-gpu:
	@echo "Starting container with GPU support..."
	docker run -it --rm --name pytorch-kan --gpus all \
	-v $(WORKSPACE_DIR):/workspace \
	-v vscode-server:/root/.vscode-server \
	-e PYTHONPATH=/workspace \
	-e PYTHONUNBUFFERED=1 \
	--network host \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	$(IMAGE_NAME_CUDA)

container-cpu:
	@echo "Starting container (CPU only)..."
	docker run -it --rm --name pytorch-kan \
	-v $(WORKSPACE_DIR):/workspace \
	-v vscode-server:/root/.vscode-server \
	-e PYTHONPATH=/workspace \
	-e PYTHONUNBUFFERED=1 \
	--network host \
	$(IMAGE_NAME)

# Setup virtual environment inside container
setup-venv:
    python3 -m venv /workspace/venv && \
    . /workspace/venv/bin/activate && \
    pip install --no-cache-dir -e '.[dev]'

# Optional: Clean up resources when done
clean: clean-containers clean-images

clean-containers:
	@echo "Cleaning up containers..."
	docker container prune -f

clean-images:
	@echo "Cleaning up images..."
	docker image prune -f
	docker rmi $(IMAGE_NAME)

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Virtual environment created. Activate with 'source $(VENV_NAME)/bin/activate'"

# Install required packages in the virtual environment
requirements: venv
	. ./$(VENV_NAME)/bin/activate && $(VENV_NAME)/bin/pip install -e '.[dev]'
	@echo "Project (dev extras) installed in virtual environment"

# Clean virtual environment
clean-venv:
	rm -rf $(VENV_NAME)
	@echo "Virtual environment removed"

# Run main.py with the virtual environment
run:
	. ./$(VENV_NAME)/bin/activate && $(VENV_NAME)/bin/python tutorials/main.py

# Display usage help
help:
	@echo "Usage:"
	@echo "  make setup         - Create required Docker volumes"
	@echo "  make build        - Build CPU-only Docker image"
	@echo "  make build-no-cache - Build Docker image without cache"
	@echo "  make build-cuda   - Build CUDA Docker image (uses Dockerfile.cuda)"
	@echo "  make container-gpu - Run container with GPU support"
	@echo "  make container-cpu - Run container (CPU only)"
	@echo "  make dist         - Build sdist and wheel"
	@echo "  make publish      - Upload distribution to PyPI (requires TWINE credentials)"
	@echo "  make clean        - Clean up all resources"

# Packaging and publishing
dist:
	python -m build

publish: dist
	twine upload dist/*

.PHONY: setup build build-no-cache build-cuda container-gpu container-cpu clean clean-containers clean-images setup-venv venv requirements clean-venv run help
