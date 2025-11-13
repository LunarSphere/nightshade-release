# FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
# ENV DEBIAN_FRONTEND=noninteractive
# ENV TOKENIZERS_PARALLELISM=false

# WORKDIR /app
# COPY . /app

# RUN python3.11 -m venv /opt/venv \
#  && /opt/venv/bin/python -m ensurepip --upgrade \
#  && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# ENV PATH="/opt/venv/bin:$PATH"

# # Install matched torch/torchvision/torchaudio for CUDA 12.8
# RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio

# # Other Python deps
# # NOTE: requirements.txt should NOT pin torch/torchvision or triton
# RUN pip install --no-cache-dir awscli boto3 \
#  && pip install --no-cache-dir -r requirements.txt

# CMD ["bash", "/app/Data_Pipeline/run_docker.bash"]

#local build and run
#docker build -t nightshade:local .
#docker run -it -v ~/.aws:/root/.aws:ro --name nightshade nightshade:local

#aws pull and run
#docker pull 782977425966.dkr.ecr.us-east-1.amazonaws.com/nightshade:latest
#docker run -it --gpus all   -v ~/.aws:/root/.aws:ro   -v /data:/data   --name nightshade   782977425966.dkr.ecr.us-east-1.amazonaws.com/nightshade:latest

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY . /app

# Optional: small system libs useful for PIL/IO
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates libglib2.0-0 libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Isolated venv
RUN python3.11 -m venv /opt/venv \
 && /opt/venv/bin/python -m ensurepip --upgrade \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel

ENV PATH="/opt/venv/bin:$PATH"

# Install CUDA 12.8-matched torch stack (do NOT install torch/torchvision via requirements.txt)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

# App dependencies
# Make sure your requirements.txt does not include torch/torchvision
# Also remove duplicate/conflicting packages (keep openai-clip OR clip; keep umap-learn, drop umap)
RUN pip install --no-cache-dir awscli boto3 \
 && pip install --no-cache-dir -r requirements.txt

# Prefetch BLIP weights to avoid cold-start download
RUN python - <<'PY'
from transformers import BlipProcessor, BlipForConditionalGeneration
BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("BLIP cached")
PY

# Ensure runner scripts are executable
RUN chmod +x /app/Data_Pipeline/*.bash

CMD ["bash", "/app/Data_Pipeline/run_docker.bash"]