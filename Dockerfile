# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
# avoid interaction
ENV DEBIAN_FRONTEND=noninteractive

# COPY REPO TO WORKDIR
WORKDIR /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    unzip \
    awscli \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
# - keep numpy < 2 as your original file did
# - explicitly install torchvision from the PyTorch CUDA wheel index so it matches the preinstalled torch
#   (this avoids the "operator torchvision::nms does not exist" runtime error caused by mismatched builds)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel boto3 "numpy<2" \
 && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 torchvision \
 && pip install --no-cache-dir -r requirements.txt

# Ensure that script runs on container start
CMD ["bash", "/app/Data_Pipeline/run_docker.bash"]