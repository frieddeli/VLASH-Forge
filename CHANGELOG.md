## [2026-04-29] Cloud deployment infrastructure — Docker, Singularity, Kubernetes + testing checklist

### Changed
- `pyproject.toml`: removed unused `flask` dependency; moved `reachy2_sdk` to `[robot]` optional extra; removed `cuda-toolkit` from pixi deps (provided by NVIDIA base image) to avoid conda cross-compiler sysroot with old kernel headers
- `Dockerfile`: switched base image to `nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu24.04` — this eliminates the evdev build failure by preventing pixi from installing the conda cross-compiler whose bundled sysroot lacks modern kernel input codes; added `CUDA_HOME` and updated `LD_LIBRARY_PATH` for deepspeed/bitsandbytes CUDA extension builds

### Added
- `Dockerfile`: single-stage ubuntu:24.04 image; pixi owns Python 3.12 + CUDA 12.6 toolkit from conda-forge; env baked at build time via `pixi install --frozen`
- `docker-entrypoint.sh`: HuggingFace login, PBSpro UUID→integer CUDA_VISIBLE_DEVICES normalisation, accelerate launch wrapper
- `.dockerignore`: excludes `.pixi/`, outputs, git history, report; keeps `pixi.lock` for reproducible builds
- `docker-compose.yaml`: single-node multi-GPU setup with persistent HF cache and output volumes, NCCL-compatible `ipc: host` and `network_mode: host`
- `examples/train/pi05/cloud.yaml`: cloud-agnostic training config using `${DATASET_REPO_ID}` env var placeholder and `/workspace/outputs` output dir
- `singularity.def`: bootstraps from Docker image for HPC (rootless, `--nv` GPU passthrough); sets HF_HOME default to scratch
- `k8s/training-job.yaml`: Kubernetes Job manifest with GPU resource limits, PVC mounts for model cache and outputs, and HF token secret reference
