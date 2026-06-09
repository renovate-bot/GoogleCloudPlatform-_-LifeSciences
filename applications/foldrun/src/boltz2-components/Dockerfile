FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install base dependencies and python
RUN apt-get update && apt-get install -y \
    wget \
    git \
    python3 \
    python3-pip \
    hmmer \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from locked requirements with hashes
COPY requirements.txt .
RUN pip3 install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements.txt

# No GCS libraries needed — all I/O via NFS mounts and KFP artifacts,
# consistent with the AF2 container.

# Do NOT bake in weights — they come from NFS at runtime
# Do NOT bake in databases — they come from NFS
