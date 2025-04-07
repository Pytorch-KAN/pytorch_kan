# Makefile for PyTorch KAN project

# Variables
IMAGE_NAME = pytorch-kan
CONTAINER_NAME = pytorch-kan
WORKSPACE_DIR = $(shell pwd)
VENV_NAME = kan_venv
PYTHON = python3
POETRY = poetry
SPHINX_BUILD = $(POETRY) run sphinx-build
SPHINX_AUTOBUILD = $(POETRY) run sphinx-autobuild

# Main targets
.PHONY: all clean docs docs-clean docs-live test lint type setup build build-no-cache container clean-containers clean-images setup-venv venv requirements clean-venv run poetry-install poetry-update poetry-run export-requirements help

all: test lint type docs

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

# Documentation targets
docs: 
	@echo "Building Sphinx documentation..."
	$(SPHINX_BUILD) -b html docs/source docs/build/html
	@echo "Documentation built successfully. Open docs/build/html/index.html in your browser."

docs-clean:
	@echo "Cleaning documentation build..."
	rm -rf docs/build
	@echo "Documentation build cleaned."

docs-live:
	@echo "Starting live preview of documentation..."
	$(SPHINX_AUTOBUILD) docs/source docs/build/html --open-browser

# Testing and quality targets
test:
	$(POETRY) run pytest tests

lint:
	$(POETRY) run black src tests
	$(POETRY) run isort src tests
	$(POETRY) run flake8 src tests

type:
	$(POETRY) run mypy src

# Project installation and setup
install:
	$(POETRY) install

dev-install:
	$(POETRY) install --with dev

update:
	$(POETRY) update

# Generate requirements.txt from Poetry dependencies
requirements:
	$(POETRY) export -f requirements.txt --output requirements.txt

# Clean up
clean: docs-clean clean-containers clean-images
	rm -rf __pycache__ .pytest_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

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

# Help
help:
	@echo "Available targets:"
	@echo "  all            - Run tests, linting, type checking, and build docs"
	@echo "  setup          - Create required Docker volumes"
	@echo "  build          - Build Docker image"
	@echo "  build-no-cache - Build Docker image without cache"
	@echo "  container      - Run container with GPU support"
	@echo "  docs           - Build Sphinx documentation"
	@echo "  docs-clean     - Clean documentation build"
	@echo "  docs-live      - Start live preview of documentation"
	@echo "  test           - Run tests"
	@echo "  lint           - Run code style checks"
	@echo "  type           - Run type checking"
	@echo "  install        - Install dependencies using Poetry"
	@echo "  dev-install    - Install development dependencies using Poetry"
	@echo "  update         - Update dependencies using Poetry"
	@echo "  requirements   - Generate requirements.txt from Poetry"
	@echo "  clean          - Clean build artifacts and cache files"
