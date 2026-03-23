# Use Python 3.13.1 as base image
FROM python:3.13.1-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the patch scripts
COPY patches/ patches/

# Run the patches
RUN python patches/patch_libraries.py

# Optional direct download URL for Dragon Jump Linux build archive (dragon-jump-linux.zip).
# Example: --build-arg DRAGON_JUMP_URL="https://<signed-itch-download-url>"
ARG DRAGON_JUMP_URL=""

# Download and extract Dragon Jump when URL is provided.
# Expected archive: dragon-jump-linux.zip (contains DragonJump.x86_64).
RUN mkdir -p /opt/dragon-jump && \
    if [ -n "$DRAGON_JUMP_URL" ]; then \
        echo "Downloading Dragon Jump from provided URL..." && \
        curl -fL "$DRAGON_JUMP_URL" -o /tmp/dragon-jump-linux.zip && \
        unzip -q /tmp/dragon-jump-linux.zip -d /opt/dragon-jump && \
        rm -f /tmp/dragon-jump-linux.zip && \
        bin_path="$(find /opt/dragon-jump -type f -name 'DragonJump.x86_64' | head -n 1)" && \
        test -n "$bin_path" && chmod +x "$bin_path"; \
    else \
        echo "DRAGON_JUMP_URL not set; skipping game download."; \
    fi

# Default game location inside Docker image.
ENV DRAGON_JUMP_DIR=/opt/dragon-jump

# Expose Jupyter notebook port
EXPOSE 8888

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=plaiground

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
