# Variables
IMAGE_NAME = pytorch-kan
CONTAINER_NAME = pytorch-kan
WORKSPACE_DIR = $(shell pwd)

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

build-no-cache:
	docker build --no-cache -t $(IMAGE_NAME) .

# Run the container with volume
container:
	docker run --rm --name $(CONTAINER_NAME) --gpus all -v $(WORKSPACE_DIR):/workspace $(IMAGE_NAME) bash

# Remove the image
remove-image:
	docker rmi $(IMAGE_NAME)

# Clean up: remove all stopped containers and unused images
clean:
	docker container prune -f
	docker image prune -f

.PHONY: build container build-run remove-image clean
