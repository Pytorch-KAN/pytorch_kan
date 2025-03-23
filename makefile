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
	-v $(WORKSPACE_DIR):/workspace \
	-v vscode-server:/root/.vscode-server \
	-e PYTHONPATH=/workspace \
	-e PYTHONUNBUFFERED=1 \
	--network host \
	--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
	--privileged \
	$(IMAGE_NAME)

# Add vscode-server volume
vscode-volume:
	docker volume create vscode-server

# Remove the image
remove-image:
	docker rmi $(IMAGE_NAME)

# Clean up: remove all stopped containers and unused images
clean:
	docker container prune -f
	docker image prune -f

.PHONY: build container build-run remove-image clean vscode-volume
