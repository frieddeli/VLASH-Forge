<!-- markdownlint-disable MD001 MD041 -->

<p align="center">
  <picture>
    <img alt="VLASH" src="assets/logo.png" width=40%>
  </picture>
</p>
<h3 align="center">
Easy-to-use VLA deployment, fast to react, smooth in motion.
</h3>

<p align="center">
    <a href="https://arxiv.org/abs/2512.01031"><b>Paper</b></a>
    &nbsp;|&nbsp;
    <a href="https://youtu.be/PLACEHOLDER"><b>Demo Video</b></a>
</p>

---

## Group Members

| Name | Student ID | Email | Contribution |
|------|-----------|-------|--------------|
| TBD  | TBD       | TBD   | TBD          |

---

## About

VLASH is an efficient and easy-to-use framework for VLAs fine-tuning and inference.

VLASH is efficient through:

- Asynchronous inference for **fast reaction and smooth motion** in real-time (**>30Hz inference frequency** for $\pi_{0.5}$ on RTX 5090)
- Future-state-awareness to enable **stable asynchronous VLA inference without overhead**
- Action quantization for **faster robot execution speed**
- LoRA with shared observation encoding for **efficient fine-tuning on consumer GPUs**

VLASH is easy to use with:

- **Seamless integration with [LeRobot](https://github.com/huggingface/lerobot)** datasets (v2.1, v3.0), models and robots
- Simple YAML-based configuration system
- Support for various policy architectures (e.g., $\pi_{0.5}$, $\pi_0$, ...)
- Easy deployment on real robot hardware
- **Docker and Singularity containers** for reproducible cloud and HPC deployment

## Demo

[![Watch the video](assets/demo-first-frame.png)](https://www.youtube.com/watch?v=IgN7CNicJS8)

---

## First-time setup

Before running training you need three things: a HuggingFace account with a
token, access to the gated base model, and your dataset uploaded to HF Hub.
This is a one-time process per account.

### 1. HuggingFace token

Create an account at [huggingface.co](https://huggingface.co) if you don't have one.
Generate a token with **write** permission at
`huggingface.co/settings/tokens` — write access is needed to push trained
checkpoints back to the Hub.

Keep this token as `HF_TOKEN` in your environment. Never commit it to the repo.

### 2. Accept the base model licenses

VLASH fine-tunes PaliGemma-based models that are gated — you must accept the
license on the HF website once before your token can download the weights.

Visit each model page and click **Agree and access repository**:

- **π₀.₅** — `huggingface.co/lerobot/pi05_base`
- **π₀** — `huggingface.co/lerobot/pi0_base`

Acceptance is per-account and propagates immediately. The container downloads
the weights automatically on first run using your `HF_TOKEN`; subsequent runs
use the cache in `$SCRATCH/.cache/huggingface`.

### 3. Upload your dataset

Your dataset must be in [LeRobot format](https://github.com/huggingface/lerobot)
(`data/`, `videos/`, `meta/` folders with a `meta/info.json`).

```bash
# Authenticate (once per machine)
huggingface-cli login --token $HF_TOKEN

# Create the dataset repo
huggingface-cli repo create your-dataset-name --type dataset --private

# Upload — preserves directory structure exactly
huggingface-cli upload your-hf-username/your-dataset-name \
  /path/to/local/lerobot/dataset/ \
  --repo-type dataset
```

Set `DATASET_REPO_ID=your-hf-username/your-dataset-name` when running training.
The container downloads it automatically via `HF_TOKEN`.

**Team / shared datasets:** create a [HuggingFace organisation](https://huggingface.co/organizations/new),
push under `your-org/your-dataset-name`, and invite collaborators at
`huggingface.co/your-org` → Settings → Members.

### 4. (Optional) Push trained checkpoints back to Hub

Set `push_to_hub: true` and `repo_id: your-org/your-model` in your training
config to automatically upload the final LoRA checkpoint after training.
This is the recommended way to persist checkpoints beyond the scratch disk
lifetime on HPC clusters.

---

## Getting Started

### Storage setup (all environments)

Every environment needs a persistent directory that the container mounts as `/scratch`.
This is where model weights, datasets, and checkpoints all live — nothing important is written
inside the container itself.

Set `SCRATCH` to wherever your persistent storage is before running:

```bash
# HPC (NSCC ASPIRE / SLURM)
export SCRATCH=/scratch/users/ntu/<your-id>

# AWS — EBS volume mounted at /mnt/ebs
export SCRATCH=/mnt/ebs

# GCP — Persistent Disk mounted at /mnt/pd
export SCRATCH=/mnt/pd

# GCP — GCS bucket via FUSE (mount first: gcsfuse your-bucket /mnt/gcs)
export SCRATCH=/mnt/gcs

# Local workstation
export SCRATCH=$HOME/vlash-scratch && mkdir -p $SCRATCH
```

Inside `/scratch` the layout is always:
```
$SCRATCH/
  .cache/huggingface/   ← base model weights (pi0.5, PaliGemma) — ~10 GB, downloaded once
  .cache/lerobot/       ← dataset videos and parquet files
  outputs/              ← training checkpoints
```

---

### Option A — Unified launcher (Docker or Singularity)

`scripts/train.sh` detects whether `vlash.sif` exists and launches via Singularity (HPC)
or Docker (cloud/local) automatically. It always binds `$SCRATCH` to `/scratch`.

```bash
export SCRATCH=/your/persistent/storage
export HF_TOKEN=<your_hf_token>
export DATASET_REPO_ID=<your-org/your-dataset>

# Single GPU
./scripts/train.sh examples/train/pi05/cloud.yaml

# Multi-GPU
NUM_GPUS=4 ./scripts/train.sh examples/train/pi05/cloud.yaml
```

**Prerequisites:** Docker with [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed, or Singularity with `vlash.sif` present in the working directory.

Build the Docker image if you haven't already (no GPU needed):
```bash
docker build -t vlash:latest .
```

Pull the Singularity image on HPC:
```bash
singularity pull vlash.sif docker://frieddeli/vlash:latest
# For faster HPC transfer, build locally and scp:
# scp vlash.sif user@nscc-server:${SCRATCH}/vlash.sif
```

> **First-run note:** DeepSpeed and bitsandbytes compile CUDA extensions on the first training run (~1–3 minutes). This is a one-time overhead cached in `$SCRATCH/.cache`.

### Option B — Docker Compose (single-node multi-GPU)

```bash
export SCRATCH=/your/persistent/storage
export HF_TOKEN=<your_hf_token>
export DATASET_REPO_ID=<your-org/your-dataset>
export NUM_GPUS=4

docker-compose up
```

### Option C — Kubernetes

See [k8s/training-job.yaml](k8s/training-job.yaml) for a complete Kubernetes Job manifest.
PersistentVolumeClaims in the manifest replace the `/scratch` bind-mount — the underlying
storage (EBS, Filestore, GCS) is configured in the PVC spec, invisible to the container.

```bash
kubectl create secret generic hf-secret --from-literal=token=<YOUR_HF_TOKEN>
kubectl apply -f k8s/training-job.yaml
kubectl logs -f job/vlash-train
```

---

## Multi-GPU Training

VLASH uses [HuggingFace Accelerate](https://huggingface.co/docs/accelerate) with DeepSpeed ZeRO-2 for distributed training. The `deepspeed_config.yaml` at the repo root configures ZeRO-2 (optimizer state + gradient sharding), bf16 mixed precision, and gradient clipping.

The number of GPUs is controlled by the `NUM_GPUS` environment variable (default: 1). When using Docker Compose, set `NUM_GPUS=4` in your environment.

**ZeRO-2 vs DDP:** ZeRO-2 shards optimizer states and gradients across GPUs, roughly halving per-GPU memory vs vanilla DDP for the same batch size. For pi0.5 (~3B parameters) on 4×A100 this enables `batch_size=1` with `grad_accum_steps=8` without CPU offloading.

---

## Local Installation (without Docker)

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Install the pinned environment
pixi install

# Verify
pixi run python -c "import vlash; print('OK')"
```

### Quick Examples

**Fine-tune a VLA policy for your task:**

```bash
vlash train examples/train/pi05/async_lora.yaml
```

**Distributed training on 4 GPUs:**

```bash
accelerate launch \
  --config_file deepspeed_config.yaml \
  --num_processes 4 \
  -m vlash.train examples/train/pi05/cloud.yaml
```

**Run async inference on a robot:**

```bash
vlash run examples/inference/async.yaml
```

**Run async inference with 2x speedup:**
```bash
vlash run examples/inference/async.yaml --action_quant_ratio=2
```

---

## TODO
- [x] LoRA fine-tuning for $\pi_{0.5}$, $\pi_0$ under 12G GPU memory
- [ ] QLoRA fine-tuning for $\pi_{0.5}$, $\pi_0$ under 8G GPU memory
- [x] Efficient fine-tuning with shared observation
- [x] DeepSpeed ZeRO-2 distributed training
- [x] Docker and Singularity containers for cloud/HPC deployment

## Acknowledgment

This project is built upon the following excellent open-source projects: [LeRobot](https://github.com/huggingface/lerobot), [PEFT](https://github.com/huggingface/peft), [DeepSpeed](https://github.com/microsoft/DeepSpeed), [pixi](https://pixi.sh).

## License

Apache 2.0
