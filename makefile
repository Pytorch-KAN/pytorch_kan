# Variables
IMAGE_NAME = pytorch-kan
CONTAINER_NAME = pytorch-kan
DOCKER_REGISTRY = your-docker-registry
WORKSPACE_DIR = $(shell pwd)

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the container with volume
container:
	docker run --rm --name $(CONTAINER_NAME) --gpus all -v $(WORKSPACE_DIR):/workspace $(IMAGE_NAME) bash

# Build and run
build-run: build container

# Push the image to the registry
push:
	docker push $(DOCKER_REGISTRY)/$(IMAGE_NAME)

# Pull the image from the registry
pull:
	docker pull $(DOCKER_REGISTRY)/$(IMAGE_NAME)

# Remove the image
remove-image:
	docker rmi $(IMAGE_NAME)

# Clean up: remove all stopped containers and unused images
clean:
	docker container prune -f
	docker image prune -f

# Check if the image has been modified and needs to be pushed
check-and-push:
	@if [ -n "$$(docker images -q $(IMAGE_NAME))" ]; then \
		if [ -n "$$(docker images --format '{{.ID}}' $(IMAGE_NAME))" != "$$(docker images --format '{{.ID}}' $(DOCKER_REGISTRY)/$(IMAGE_NAME))" ]; then \
			echo "Image has been modified. Pushing to registry..."; \
			make push; \
		else \
			echo "Image is up to date. No push needed."; \
		fi \
	else \
		echo "Image not found locally. Building and pushing..."; \
		make build push; \
	fi

.PHONY: build container build-run push pull remove-image clean check-and-push
