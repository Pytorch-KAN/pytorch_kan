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
	docker run -it --rm --name pytorch-kan --gpus all \
	-v /home/fabio/Documents/GitHub/Pytorch_KAN/pytorch_kan:/workspace \
	-e PYTHONPATH="/workspace:$$PYTHONPATH" \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	pytorch-kan bash

# Remove the image
remove-image:
	docker rmi $(IMAGE_NAME)

# Clean up: remove all stopped containers and unused images
clean:
	docker container prune -f
	docker image prune -f

.PHONY: build container build-run remove-image clean
