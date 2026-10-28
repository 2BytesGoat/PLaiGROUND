# Use Python 3.13.1 as base image
FROM python:3.13.1-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    unzip \
    linux-headers-generic \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install torch CPU
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy project metadata and package
COPY pyproject.toml poetry.lock README.md ./
COPY src ./src

# Install the project and dependencies
RUN pip install --no-cache-dir -e .

# Install jupyterlab
RUN pip install jupyterlab

# Expose Jupyter notebook port
EXPOSE 8888

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=plaiground

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
