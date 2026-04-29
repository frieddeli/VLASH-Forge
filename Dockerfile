# CUDA 12.6 devel image on Ubuntu 24.04:
# - Provides CUDA toolkit so pixi doesn't need to install cuda-toolkit from
#   conda-forge, which would pull in a conda cross-compiler with an old bundled
#   sysroot. Without that cross-compiler, pip/uv uses the system gcc which reads
#   Ubuntu 24.04 kernel headers (linux-libc-dev 6.8) — those define all the
#   input codes evdev needs to compile.
FROM nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu24.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System packages needed for video decoding and Python build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    build-essential \
    libgl1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install pixi — it owns Python 3.12, CUDA 12.6 toolkit, and all ML deps
RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$PATH"

WORKDIR /workspace

# Copy project metadata; pixi will generate lockfile fresh during build
COPY pyproject.toml ./

# Copy source code and config
COPY vlash/ vlash/
COPY benchmarks/ benchmarks/
COPY examples/ examples/
COPY deepspeed_config.yaml ./

# Resolve and install all Python/ML dependencies via pixi.
# CUDA toolkit is already in the base image so pip uses the system gcc
# (not a conda cross-compiler), meaning evdev compiles cleanly.
RUN pixi install

# Activate the pixi environment by putting its bin on PATH.
# CUDA_HOME points to the toolkit already in the base image so DeepSpeed and
# bitsandbytes can find nvcc and libcudart when building CUDA extensions.
ENV PATH="/root/.pixi/envs/default/bin:$PATH"
ENV LD_LIBRARY_PATH="/root/.pixi/envs/default/lib:/usr/local/cuda/lib64"
ENV CUDA_HOME="/usr/local/cuda"

# DeepSpeed and bitsandbytes compile CUDA extensions lazily on first training
# run (~1-3 min). TORCH_CUDA_ARCH_LIST covers V100, A100, A10, 4090, H100.
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0"

# Prevent pixi from trying to update the lockfile at runtime
ENV PIXI_FROZEN=1

COPY docker-entrypoint.sh /workspace/docker-entrypoint.sh
RUN chmod +x /workspace/docker-entrypoint.sh

ENTRYPOINT ["/workspace/docker-entrypoint.sh"]
