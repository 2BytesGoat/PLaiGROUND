#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker from https://www.docker.com/products/docker-desktop/ and try again."
    exit 1
fi

DRAGON_JUMP_PAGE_URL="http://2bytesgoat.itch.io/dragon-jump"

# DRAGON_JUMP_URL must be a direct file URL to dragon-jump-linux.zip
if [ -z "${DRAGON_JUMP_URL}" ]; then
    echo "No DRAGON_JUMP_URL set. Open ${DRAGON_JUMP_PAGE_URL} and copy the direct link to dragon-jump-linux.zip."
fi

# Build the Docker image
echo "Building Docker image..."
docker build \
    --build-arg DRAGON_JUMP_URL="${DRAGON_JUMP_URL}" \
    -t jupyter-env .

# Run the Docker container with the RL environment mounted
echo "Starting Docker container with mounted RL environment..."
docker run -it --rm \
    -p 8888:8888 \
    -p 11008:11008 \
    -v "$(pwd)":/app \
    jupyter-env
