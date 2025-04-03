# Variables
IMAGE_NAME = pytorch-kan
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

# Step 3: Run the container and setup venv
container:
	@echo "Starting container with GPU support..."
	docker run -it --rm --name pytorch-kan --gpus all \
	-v $(WORKSPACE_DIR):/workspace \
	-v vscode-server:/root/.vscode-server \
	-v pytorch-venv:/workspace/venv \
	-e PYTHONPATH=/workspace \
	-e PYTHONUNBUFFERED=1 \
	--network host \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	--privileged \
	$(IMAGE_NAME)

# Local development with Poetry
poetry-install:
	poetry install

poetry-update:
	poetry update

# Generate requirements.txt from Poetry for environments that don't use Poetry
export-requirements:
	poetry export -f requirements.txt --output requirements.txt

# Run in Poetry environment
poetry-run:
	poetry run python tutorials/main.py

# Setup virtual environment inside container (legacy, using poetry inside container is preferred)
setup-venv:
	python3 -m venv /workspace/venv && \
	. /workspace/venv/bin/activate && \
	pip install --no-cache-dir -r requirements.txt

# Optional: Clean up resources when done
clean: clean-containers clean-images

clean-containers:
	@echo "Cleaning up containers..."
	docker container prune -f

clean-images:
	@echo "Cleaning up images..."
	docker image prune -f
	docker rmi $(IMAGE_NAME)

# Create virtual environment (legacy, use poetry-install instead)
venv:
	$(PYTHON) -m venv $(VENV_NAME)
	@echo "Virtual environment created. Activate with 'source $(VENV_NAME)/bin/activate'"

# Install required packages in the virtual environment (legacy, use poetry-install instead)
requirements: venv
	. ./$(VENV_NAME)/bin/activate && $(VENV_NAME)/bin/pip install -r requirements.txt
	@echo "Requirements installed in virtual environment"

# Clean virtual environment
clean-venv:
	rm -rf $(VENV_NAME)
	@echo "Virtual environment removed"

# Run main.py with the virtual environment (legacy, use poetry-run instead)
run:
	. ./$(VENV_NAME)/bin/activate && $(VENV_NAME)/bin/python tutorials/main.py

# Display usage help
help:
	@echo "Usage:"
	@echo "  make setup         - Create required Docker volumes"
	@echo "  make build        - Build Docker image"
	@echo "  make build-no-cache - Build Docker image without cache"
	@echo "  make container    - Run container with GPU support"
	@echo "  make poetry-install - Install dependencies using Poetry"
	@echo "  make poetry-update - Update dependencies using Poetry"
	@echo "  make poetry-run   - Run main.py using Poetry"
	@echo "  make export-requirements - Generate requirements.txt from Poetry dependencies"
	@echo "  make clean        - Clean up all resources"

.PHONY: setup build build-no-cache container clean clean-containers clean-images setup-venv venv requirements clean-venv run poetry-install poetry-update poetry-run export-requirements help
