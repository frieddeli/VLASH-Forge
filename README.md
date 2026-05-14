<!-- markdownlint-disable MD001 MD041 -->

<h1 align="center">VLASH Forge</h1>
<p align="center">
  <picture>
    <img alt="VLASH Forge" src="assets/logo.png" width=40%>
  </picture>
</p>
<h3 align="center">
Portable distributed VLA fine-tuning on cloud and HPC — one command to train anywhere.
</h3>

<p align="center">
    <a href="https://arxiv.org/abs/2512.01031"><b>Paper</b></a> |
    <a href="https://frieddeli.github.io/VLASH-Forge"><b>Docs</b></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.01031">
    <img src="https://img.shields.io/badge/Based%20on-arXiv%3A2512.01031-b31b1b.svg" alt="Based on arXiv:2512.01031">
  </a>
  <a href="https://hub.docker.com/r/frieddeli/vlash-forge">
    <img src="https://img.shields.io/badge/Docker-frieddeli%2Fvlash-2496ED?logo=docker&logoColor=white" alt="Docker Hub">
  </a>
  <a href="https://huggingface.co/lerobot/pi05_base">
    <img src="https://img.shields.io/badge/🤗-pi0.5%20%7C%20pi0-FFD21E" alt="HuggingFace Models">
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/CUDA-12.6-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.6">
  <img src="https://img.shields.io/badge/DeepSpeed-ZeRO--2-9B59B6" alt="DeepSpeed ZeRO-2">
  <a href="https://frieddeli.github.io/VLASH-Forge">
    <img src="https://img.shields.io/badge/Docs-GitHub%20Pages-0A0A0A?logo=githubpages&logoColor=white" alt="Documentation">
  </a>

  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

---

## About

Fine-tuning a state-of-the-art robot policy like π₀.₅ on your own task requires
multi-GPU distributed training, a carefully managed Python environment, and
infrastructure that has never been publicly released for the VLASH framework.
**This repo removes those barriers.**

It packages [VLASH](https://arxiv.org/abs/2512.01031) — a VLA fine-tuning and
deployment framework built on [LeRobot](https://github.com/huggingface/lerobot)
— into a single container image that runs on a cloud VM, an HPC cluster, or a
local workstation with one command. You bring your robot demonstrations; the
pipeline handles the rest.

**Full documentation:** [frieddeli.github.io/VLASH-Forge](https://frieddeli.github.io/VLASH-Forge)

**The intended workflow:**

```
1. Collect demonstrations on your robot  →  upload to HuggingFace Hub
2. Run:  ./scripts/train.sh config.yaml  →  fine-tuned checkpoint on HF Hub
3. Load checkpoint on your inference hardware  →  deploy
```

**What VLASH adds over standard LeRobot:**

- **Asynchronous inference** via Temporal Delay Augmentation — overlaps inference
  with execution for **29.5× lower latency** on embedded hardware
- **LoRA + shared observation encoding** — fine-tune π₀.₅ on **12 GB VRAM**
- **DeepSpeed ZeRO-2** (LoRA) and **FSDP** (full fine-tuning) distributed backends
- **Docker + Singularity** — same image on AWS, GCP, NSCC ASPIRE, or a local GPU

Validated end-to-end on a ball pick-and-place task: demonstrations collected on a
Piper arm, fine-tuned on NSCC ASPIRE, deployed on a Jetson AGX Orin at 30 Hz —
**65% task success rate** with async inference vs. 5% synchronous baseline.

---

## First-time Setup

Before running training you need three things: a HuggingFace account with a token,
access to the gated base models, and your dataset uploaded to HF Hub.
This is a one-time process per account.

### 1. HuggingFace token

Create an account at [huggingface.co](https://huggingface.co) if you don't have one.
Generate a token with **write** permission at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
— write access is needed to push trained checkpoints back to the Hub.

Keep this token as `HF_TOKEN` in your environment. Never commit it to the repo.

### 2. Accept the base model licenses

VLASH fine-tunes PaliGemma-based models that are gated — you must accept the
license on the HF website once before your token can download the weights.

Visit each model page and click **Agree and access repository**:

- **π₀.₅** — [huggingface.co/lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)
- **π₀** — [huggingface.co/lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base)

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

### 4. (Optional) Push trained checkpoints to Hub

Set `push_to_hub: true` and `repo_id: your-org/your-model` in your training config
to automatically upload the final checkpoint after training. This is the recommended
way to persist checkpoints beyond the scratch disk lifetime on HPC clusters.

---

## Getting Started

### Storage setup (all environments)

Every environment needs a persistent directory that the container mounts as `/scratch`.
Model weights, datasets, and checkpoints all live here — nothing important is written
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
  .cache/huggingface/   ← base model weights (~10 GB, downloaded once)
  .cache/lerobot/       ← dataset videos and parquet files
  outputs/              ← training checkpoints
```

### Option A — Unified launcher (Docker or Singularity)

`scripts/train.sh` detects whether `vlash.sif` exists and launches via Singularity (HPC)
or Docker (cloud/local) automatically. It always binds `$SCRATCH` to `/scratch`.

```bash
export SCRATCH=/your/persistent/storage
export HF_TOKEN=<your_hf_token>
export DATASET_REPO_ID=<your-org/your-dataset>

# Single GPU — LoRA fine-tuning (default)
./scripts/train.sh examples/train/pi05/cloud.yaml

# Multi-GPU — LoRA fine-tuning
NUM_GPUS=4 ./scripts/train.sh examples/train/pi05/cloud.yaml

# Full fine-tuning (requires 40 GB+ VRAM per GPU)
TRAIN_BACKEND=fsdp ./scripts/train.sh examples/train/pi05/cloud_full.yaml
```

**Prerequisites:** Docker with [NVIDIA Container Runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed, or Singularity with `vlash.sif` present in the working directory.

Build the Docker image if you haven't already (no GPU needed for build):

```bash
docker build -t vlash-forge:latest .
```

Pull the Singularity image on HPC:

```bash
singularity pull vlash.sif docker://frieddeli/vlash-forge:latest
# For faster HPC transfer, build locally and copy:
# scp vlash.sif user@nscc-server:${SCRATCH}/vlash.sif
```

> **First-run note:** DeepSpeed and bitsandbytes compile CUDA extensions on first use (~1–3 minutes). This is a one-time overhead cached in `$SCRATCH/.cache`.

### Option B — Docker Compose (single-node multi-GPU)

```bash
cp .env.example .env   # fill in HF_TOKEN, DATASET_REPO_ID, SCRATCH, NUM_GPUS
docker-compose up
```

### Option C — SLURM (HPC job scheduler)

```bash
export SCRATCH=/scratch/users/ntu/<your-id>
export HF_TOKEN=<your_hf_token>
export DATASET_REPO_ID=<your-org/your-dataset>

sbatch scripts/train_slurm.sbatch examples/train/pi05/cloud.yaml
```

Edit the `#SBATCH` directives at the top of [scripts/train_slurm.sbatch](scripts/train_slurm.sbatch)
to match your cluster's partition name, GPU count, and time limit.

### Option D — Kubernetes

See [k8s/training-job.yaml](k8s/training-job.yaml) for a complete Job manifest.
PersistentVolumeClaims replace the `/scratch` bind-mount — the underlying storage
(EBS, Filestore, GCS) is configured in the PVC spec, invisible to the container.

```bash
kubectl create secret generic hf-secret --from-literal=token=<YOUR_HF_TOKEN>
kubectl apply -f k8s/training-job.yaml
kubectl logs -f job/vlash-train
```

---

## Training Backends

Two backends are available, selected via the `TRAIN_BACKEND` environment variable:

| `TRAIN_BACKEND`       | Method                  | When to use                                                                          |
| ----------------------- | ----------------------- | ------------------------------------------------------------------------------------ |
| `deepspeed` (default) | DeepSpeed ZeRO-2        | LoRA fine-tuning — shards optimiser states and gradients, keeps params replicated   |
| `fsdp`                | PyTorch FSDP FULL_SHARD | Full fine-tuning — shards params + optimiser states + gradients (ZeRO-3 equivalent) |

Use `deepspeed` (the default) for LoRA. Switch to `fsdp` only when running `lora.enable: false`
in your config — full fine-tuning needs 40 GB+ VRAM per GPU and the `cloud_full.yaml` config.

Both backends are single-node only (1–N GPUs on one machine). Multi-node training is not
currently supported.

---

## GPU Requirements

| Model          | Mode                 | Min VRAM per GPU | Recommended hardware   |
| -------------- | -------------------- | ---------------- | ---------------------- |
| π₀.₅ (1.3B) | LoRA (`deepspeed`) | 12 GB            | RTX 3090 / A10G / T4   |
| π₀.₅ (1.3B) | Full (`fsdp`)      | 40 GB            | 4×A100 40GB           |
| π₀ (3B)      | LoRA (`deepspeed`) | 24 GB            | RTX 4090 / A100 40GB   |
| π₀ (3B)      | Full (`fsdp`)      | 80 GB            | 4×A100 80GB / 4×H100 |

AWS instance guide: `g5.xlarge` (1×A10G 24GB) for LoRA; `g5.12xlarge` (4×A10G) for multi-GPU LoRA;
`p4d.24xlarge` (8×A100 40GB) or `p4de.24xlarge` (8×A100 80GB) for full fine-tuning.

### Out of memory (OOM)

If training crashes with CUDA OOM, apply these fixes in order:

1. **Reduce `batch_size`** to 1 in your training config
2. **Increase `grad_accum_steps`** to maintain the same effective batch size
3. **Enable `gradient_checkpointing: true`** in the policy section (~20% slower, saves ~30% VRAM)
4. **Reduce `lora.r`** from 16 to 8 (halves LoRA parameter count)
5. **Switch to QLoRA** — set `lora.use_qlora: true` (4-bit base model, fits π₀.₅ in 8 GB)
6. **Full fine-tuning only** — enable `fsdp_activation_checkpointing: true` in `configs/fsdp_config.yaml`

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

### Quick examples

**LoRA fine-tuning (single GPU):**

```bash
export DATASET_REPO_ID=your-org/your-dataset
vlash train examples/train/pi05/cloud.yaml
```

**Distributed training on 4 GPUs:**

```bash
export DATASET_REPO_ID=your-org/your-dataset
accelerate launch \
  --config_file configs/deepspeed_config.yaml \
  --num_processes 4 \
  -m vlash.train examples/train/pi05/cloud.yaml
```

**Async inference on a robot:**

```bash
vlash run examples/inference/async.yaml
```

**Async inference with 2× action speedup:**

```bash
vlash run examples/inference/async.yaml --action_quant_ratio=2
```

---

## TODO

- [X] LoRA fine-tuning for π₀.₅, π₀ under 12 GB GPU memory
- [X] Full fine-tuning via FSDP for π₀.₅, π₀
- [X] Efficient fine-tuning with shared observation encoding
- [X] DeepSpeed ZeRO-2 distributed training
- [X] Docker and Singularity containers for cloud/HPC deployment
- [X] SLURM job script for HPC clusters
- [X] QLoRA fine-tuning for π₀.₅, π₀ under 8 GB GPU memory
- [ ] Multi-node training support

---

## Acknowledgements

This project is built upon the following open-source projects:
[LeRobot](https://github.com/huggingface/lerobot),
[PEFT](https://github.com/huggingface/peft),
[DeepSpeed](https://github.com/microsoft/DeepSpeed),
[pixi](https://pixi.sh).

## License

[Apache 2.0](LICENSE)
