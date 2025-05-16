# Use Python 3.13.1 as base image
FROM python:3.13.1-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the patch scripts
COPY patches/ patches/

# Run the patches
RUN python patches/patch_libraries.py

# Expose Jupyter notebook port
EXPOSE 8888

# Set environment variables
ENV JUPYTER_ENABLE_LAB=yes
ENV JUPYTER_TOKEN=plaiground

# Declare volume for mounting the current directory
VOLUME ["/app"]

# Start Jupyter notebook server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
