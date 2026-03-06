FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# System Installations
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential cmake curl ffmpeg git \
    libatlas-base-dev libboost-filesystem-dev libboost-graph-dev libboost-program-options-dev libboost-system-dev libboost-test-dev \
    libhdf5-dev libcgal-dev libeigen3-dev libflann-dev libfreeimage-dev libgflags-dev libglew-dev libgoogle-glog-dev \
    libmetis-dev libprotobuf-dev libqt5opengl5-dev libsqlite3-dev libsuitesparse-dev \
    nano protobuf-compiler python-is-python3 python3-dev python3-pip qtbase5-dev \
    sudo vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# CUDA Architectures:
#
# | CUDA Architecture | Example Graphics Cards                 |
# |-------------------|----------------------------------------|
# | 70                | Tesla V100                             |
# | 75                | GeForce RTX 2080, Tesla T4             |
# | 80                | GeForce RTX 3080, Tesla A100           |
# | 86                | GeForce RTX 3090                       |
# | 89                | GeForce RTX 4090                       |
# | 90                | H100                                   |
# |-------------------|----------------------------------------|

ENV CUDA_ARCHITECTURES="86;89;90"
ENV TCNN_CUDA_ARCHITECTURES="86;89;90"

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install dependencies with uv
RUN uv sync --no-dev --frozen || uv sync --no-dev

# Install tiny-cuda-nn (requires CUDA at build time)
RUN uv pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Copy source code
COPY . .

CMD ["uv", "run", "jupyter", "lab", "--ip=0.0.0.0", "--port=6789", "--allow-root", "--no-browser"]

EXPOSE 6789

# BUILD: docker build -t 6img-to-3d .
