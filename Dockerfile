# Base image with uv preinstalled on Python 3.13
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency definitions first for better layer caching
COPY pyproject.toml uv.lock ./

# Install all dependencies (including dev group: jupyter stack) from the lockfile
RUN uv sync --frozen --no-install-project

# Expose Jupyter notebook port + Godot RL socket port
EXPOSE 8888 11008

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=plaiground

# Run Jupyter Lab from the uv-managed environment
CMD ["uv", "run", "--frozen", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]