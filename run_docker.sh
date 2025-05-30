#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker from https://www.docker.com/products/docker-desktop/ and try again."
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build -t jupyter-env .

# Run the Docker container with the RL environment mounted
echo "Starting Docker container with mounted RL environment..."
docker run -it --rm \
    -p 8888:8888 \
    -p 11008:11008 \
    -v "$(pwd)":/app \
    jupyter-env
