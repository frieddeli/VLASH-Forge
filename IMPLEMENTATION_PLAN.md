# Implementation Plan: VLASH Cloud Deployment (COMP4651)

## Context

VLASH is a VLA fine-tuning framework originally validated on the NSCC ASPIRE 2A HPC cluster. The original paper's team used 4×H100 with DDP, but the DDP code was never open-sourced. This project adapts VLASH for commercial cloud by: (1) implementing DeepSpeed ZeRO-2 distributed training (already partially done), and (2) packaging everything into portable Docker/Singularity containers. The assignment requires cloud computing focus, a ≤5-page QMD report, a 5-min video, and an updated README — all due May 17.

No Docker files exist yet. The training pipeline, DeepSpeed config, LoRA support, and pixi environment are already in place.

---

## Confirmed Design Decisions

- **GPU targets:** Generic multi-GPU (CUDA arches 7.0+8.0+8.6+8.9+9.0 — covers V100, A100, A10, 4090, H100)
- **Container targets:** Docker (docker-compose) + Singularity + Kubernetes Job manifest
- **Base image:** `ubuntu:22.04` — pixi owns Python 3.12 + CUDA 12.6 toolkit from conda-forge. NVIDIA Container Runtime injects GPU drivers at container startup. No conflicting CUDA installations.
- **pixi at build time:** `pixi install` runs during `docker build`, baking the full pixi.lock-pinned environment into the image. Fast cold starts, no internet needed at runtime.
- **Model downloads:** `HF_TOKEN` env var → entrypoint logs in → HuggingFace downloads on first run and caches in `HF_HOME`. `HF_HOME` is a persistent volume mount so 10GB+ models are not re-downloaded on every container restart. Works on cloud (internet available) and NSCC compute nodes (internet confirmed available).
- **Report data:** Existing training runs available; will run inference benchmarks inside container for real latency/FPS numbers.

---

## Portability Audit Findings (pre-implementation)

Issues found and addressed in this plan:

| Issue | Severity | Fix |
|-------|----------|-----|
| `reachy2_sdk` in base deps (robot-only SDK) | Medium | Move to `[robot]` optional extra in pyproject.toml |
| `flask` dependency (unused, zero imports) | Low | Remove from pyproject.toml |
| Hardcoded NSCC paths in mytrain*.yaml examples | High | Add `examples/train/pi05/cloud.yaml` template with env var placeholders |
| DeepSpeed + bitsandbytes lazy CUDA compile on first run | Low | Document in README (1-3 min overhead, not a bug) |
| `CUDA_VISIBLE_DEVICES` UUID format on PBSpro/NSCC | Medium | Auto-normalize in entrypoint (no-op on SLURM/cloud) |

---

## Phase 0 — Dependency Cleanup

### 0.1 Update `pyproject.toml`
- Remove `flask` from base dependencies (zero imports anywhere)
- Move `reachy2_sdk` out of base deps into a new `[robot]` optional extra
- Add `[project.optional-dependencies]` section:
  ```toml
  [project.optional-dependencies]
  robot = ["reachy2_sdk"]
  ```
- Docker image installs base deps only (`pip install -e .`), not `[robot]`

### 0.2 Add `examples/train/pi05/cloud.yaml`
Cloud-agnostic training config template. Uses environment variable placeholders for all paths:
```yaml
policy:
  type: pi05
  pretrained_path: lerobot/pi05_base   # downloaded at runtime via HF_TOKEN
dataset:
  repo_id: ${DATASET_REPO_ID}          # e.g. your-hf-username/your-dataset
output_dir: /workspace/outputs/pi05_cloud
```
The NSCC-specific `mytrain_piper_test3.yaml` and `mytrain.yaml` remain untouched as cluster references.

---

## Phase 1 — Docker Infrastructure

### 1.1 `Dockerfile`
**Path:** `/home/ray/Dev/COMP4651-Project/Dockerfile`

Single-stage build — `ubuntu:22.04` base, pixi owns everything at build time:

```
Base: ubuntu:22.04
  → Install system deps: curl, git, ffmpeg, build-essential, libgl1
  → Install pixi binary (curl from install.pixi.sh)
  → COPY pyproject.toml pixi.lock ./
  → COPY vlash/ benchmarks/ examples/ deepspeed_config.yaml ./
  → pixi install --frozen  (resolves pixi.lock exactly: Python 3.12, CUDA 12.6, all ML deps)
  → DeepSpeed installs (CUDA ops compiled lazily at first training run, not here)
  → COPY docker-entrypoint.sh + chmod +x
  → ENTRYPOINT ["./docker-entrypoint.sh"]
```

**Key env vars set in Dockerfile:**
- `PATH=/root/.pixi/envs/default/bin:$PATH`
- `LD_LIBRARY_PATH=/root/.pixi/envs/default/lib:$LD_LIBRARY_PATH`
- `TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6;8.9;9.0"`
- `PIXI_FROZEN=1` — prevents pixi from attempting lockfile updates at runtime

### 1.2 `docker-entrypoint.sh`
**Path:** `/home/ray/Dev/COMP4651-Project/docker-entrypoint.sh`

```bash
#!/bin/bash
set -e

# Normalize CUDA_VISIBLE_DEVICES from UUID to integer indices.
# PBSpro (NSCC ASPIRE) sets UUIDs: "GPU-50ee0fc4-bb3d-920c-8039-da7054e1496b"
# NCCL and DeepSpeed require integer indices (0,1,2...).
# Safe no-op on SLURM and commercial cloud (already integers or unset).
if [[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]] || [[ "$CUDA_VISIBLE_DEVICES" == *,GPU-* ]]; then
    N=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N - 1)))
    echo "[entrypoint] Normalized CUDA_VISIBLE_DEVICES → $CUDA_VISIBLE_DEVICES"
fi

# Log into HuggingFace (required for gated models like pi0.5).
# Models are cached in HF_HOME — mount a persistent volume there to avoid
# re-downloading 10GB+ weights on every container restart.
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null
fi

exec pixi run accelerate launch \
    --config_file /workspace/deepspeed_config.yaml \
    --num_processes "${NUM_GPUS:-1}" \
    -m vlash.train "$@"
```

**HPC scheduler compatibility:**
- **PBSpro (NSCC ASPIRE):** UUID `CUDA_VISIBLE_DEVICES` → normalized to integers automatically
- **SLURM (~90% of global HPC):** Already integers → UUID block skipped, no effect
- **Commercial cloud:** Unset or integers → same, no effect
- UUID format is a PBSpro-specific behaviour, not universal. Container is portable across all targets without modification.

**Env vars:**
- `HF_TOKEN` — HuggingFace token (required for gated models)
- `HF_HOME` — path to HF cache; mount a persistent volume here (default: `~/.cache/huggingface`)
- `NUM_GPUS` — number of GPUs to use (default: 1; auto-detected by vlash CLI)
- `WANDB_API_KEY` — optional W&B logging

### 1.3 `.dockerignore`
**Path:** `/home/ray/Dev/COMP4651-Project/.dockerignore`

Excludes: `.pixi/`, `__pycache__/`, `*.egg-info/`, `outputs/`, `.git/`, `*.pyc`, `report/`

`pixi.lock` must **not** be excluded — copied into image so `pixi install --frozen` resolves the exact pinned environment.

### 1.4 `docker-compose.yaml`
**Path:** `/home/ray/Dev/COMP4651-Project/docker-compose.yaml`

Single-node multi-GPU setup:
```yaml
services:
  train:
    build: .
    image: vlash:latest
    runtime: nvidia
    environment:
      - HF_TOKEN=${HF_TOKEN}
      - HF_HOME=/hf_cache
      - NUM_GPUS=${NUM_GPUS:-4}
      - WANDB_API_KEY=${WANDB_API_KEY:-}
    volumes:
      - ${HF_CACHE_PATH:-./hf_cache}:/hf_cache      # persistent model cache
      - ${DATASET_PATH:-./data}:/data
      - ${OUTPUT_PATH:-./outputs}:/workspace/outputs
    command: examples/train/pi05/cloud.yaml
    ipc: host          # required for NCCL shared memory
    network_mode: host  # required for multi-GPU NCCL
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 1.5 Update `deepspeed_config.yaml`
Add human-readable comments explaining each field for non-technical users. The `num_processes` field is overridden at runtime by the entrypoint's `--num_processes` flag, so it serves as a documentation default only.

---

## Phase 2 — Singularity/HPC Support

### 2.1 `singularity.def`
**Path:** `/home/ray/Dev/COMP4651-Project/singularity.def`

Bootstrap from the Docker image — single source of truth, no duplicate build logic:

```
Bootstrap: docker
From: vlash:latest

%post
    mkdir -p /scratch /data

%environment
    export HF_HOME=${HF_HOME:-/scratch/.cache/huggingface}

%runscript
    exec /workspace/docker-entrypoint.sh "$@"
```

Usage on NSCC:
```bash
singularity build vlash.sif singularity.def
singularity run --nv \
    -B /scratch/users/ntu/m230060:/scratch \
    vlash.sif examples/train/pi05/cloud.yaml
```

The entrypoint handles UUID `CUDA_VISIBLE_DEVICES` normalization automatically — no manual `export CUDA_VISIBLE_DEVICES=0,1` needed in the PBS job script.

---

## Phase 3 — Kubernetes

### 3.1 `k8s/training-job.yaml`
**Path:** `/home/ray/Dev/COMP4651-Project/k8s/training-job.yaml`

Kubernetes Job for GPU node pools on commercial cloud clusters:
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: vlash-train
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: train
          image: vlash:latest
          args: ["examples/train/pi05/cloud.yaml"]
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef: {name: hf-secret, key: token}
            - name: HF_HOME
              value: /hf_cache
            - name: NUM_GPUS
              value: "4"
          resources:
            limits:
              nvidia.com/gpu: 4
          volumeMounts:
            - name: hf-cache
              mountPath: /hf_cache
            - name: outputs
              mountPath: /workspace/outputs
      volumes:
        - name: hf-cache
          persistentVolumeClaim: {claimName: hf-cache-pvc}
        - name: outputs
          persistentVolumeClaim: {claimName: outputs-pvc}
```

---

## Phase 4 — Documentation

### 4.1 Update `README.md`
Add sections:
- **Group Members** table (name, student ID, email, contribution) — fill when info available
- **Cloud Deployment** quickstart with `docker-compose up`
- **Multi-GPU Training** instructions
- **HPC / Singularity** one-liner for NSCC
- **First-run note:** DeepSpeed and bitsandbytes compile CUDA ops on first training run (~1-3 min overhead — not a crash)
- YouTube video link placeholder

### 4.2 `report/report.qmd`
**Path:** `/home/ray/Dev/COMP4651-Project/report/report.qmd`

Structure (≤5 pages, single column, 10pt font):
1. **Introduction** (~0.5p): Problem — HPC-only VLA fine-tuning, missing DDP code; goal — cloud-portable distributed training
2. **Background** (~0.5p): VLASH, pi0.5 architecture, DeepSpeed ZeRO-2 vs DDP
3. **System Design** (~1.5p): Architecture (Docker → Accelerate → DeepSpeed ZeRO-2 → VLASH training loop), LoRA config, dataset pipeline, container portability design
4. **Evaluation** (~2p): Inference latency benchmark (FPS, latency percentiles from container), training observations, container image size and startup overhead
5. **Conclusion** (~0.5p): Reproducibility, cloud/HPC portability, future work (QLoRA, multi-node K8s)

Quarto frontmatter:
```yaml
---
title: "Distributed VLA Fine-Tuning on the Cloud: Dockerizing VLASH with DeepSpeed"
format:
  pdf:
    fontsize: 10pt
    geometry: margin=1in
---
```

---

## Phase 5 — Verification

1. `docker build -t vlash:latest .` — must complete without GPU
2. `docker run --rm --gpus all vlash:latest pixi run python -c "import torch; print(torch.cuda.is_available())"` → must print `True`
3. `docker run --rm --gpus device=0 -e HF_TOKEN=... vlash:latest examples/train/pi05/cloud.yaml --steps=10` — smoke test
4. `NUM_GPUS=2 docker-compose up` — verify NCCL initializes across 2 GPUs
5. `singularity build vlash.sif singularity.def` — HPC image build
6. Run inference benchmark inside container → capture FPS/latency for report

---

## File Change Summary

| File | Action |
|------|--------|
| `pyproject.toml` | Update — remove `flask`, move `reachy2_sdk` to `[robot]` optional extra |
| `examples/train/pi05/cloud.yaml` | Create — portable cloud training config |
| `Dockerfile` | Create |
| `docker-entrypoint.sh` | Create |
| `.dockerignore` | Create |
| `docker-compose.yaml` | Create |
| `deepspeed_config.yaml` | Minor update — add explanatory comments |
| `singularity.def` | Create |
| `k8s/training-job.yaml` | Create |
| `README.md` | Update — group info, Docker/Singularity quickstart |
| `report/report.qmd` | Create |
| `CHANGELOG.md` | Create (required by AGENTS.md) |

---

## Remaining Open Item

- **Group member info** — names, student IDs, and emails needed for README. Provide when ready.
