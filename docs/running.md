# Running Training

## Prerequisites

- `SCRATCH`, `HF_TOKEN`, and `DATASET_REPO_ID` set in your environment
  (see [Storage Setup](storage.md) and [First-time Setup](setup.md))
- Docker image built or Singularity image pulled

```bash
# Docker
docker build -t vlash:latest .

# Singularity (HPC)
singularity pull vlash.sif docker://frieddeli/vlash:latest
```

## Option A — Unified launcher

`scripts/train.sh` detects whether `vlash.sif` is present and launches via
Singularity (HPC) or Docker (cloud/local) automatically.

```bash
export SCRATCH=/your/persistent/storage
export HF_TOKEN=hf_xxx
export DATASET_REPO_ID=your-org/your-dataset

# Single GPU — LoRA (default)
./scripts/train.sh examples/train/pi05/cloud.yaml

# Multi-GPU — LoRA
NUM_GPUS=4 ./scripts/train.sh examples/train/pi05/cloud.yaml

# Full fine-tuning (needs 40 GB+ VRAM per GPU)
TRAIN_BACKEND=fsdp ./scripts/train.sh examples/train/pi05/cloud_full.yaml
```

!!! info "First-run overhead"
    DeepSpeed and bitsandbytes compile CUDA extensions on first use (~1–3 minutes).
    This is cached in `$SCRATCH/.cache` and does not repeat on subsequent runs.

## Option B — Docker Compose (multi-GPU)

```bash
cp .env.example .env   # fill in HF_TOKEN, DATASET_REPO_ID, SCRATCH, NUM_GPUS
docker-compose up
```

## Option C — SLURM job

```bash
export SCRATCH=/scratch/users/ntu/<your-id>
export HF_TOKEN=hf_xxx
export DATASET_REPO_ID=your-org/your-dataset

sbatch scripts/train_slurm.sbatch examples/train/pi05/cloud.yaml
```

Edit the `#SBATCH` directives at the top of `scripts/train_slurm.sbatch` to match
your cluster's partition name, GPU count, and time limit.

## Option D — Kubernetes

```bash
kubectl create secret generic hf-secret --from-literal=token=<YOUR_HF_TOKEN>
kubectl apply -f k8s/training-job.yaml
kubectl logs -f job/vlash-train
```

## Monitoring training

Enable W&B logging by setting `wandb.enable: true` in your config and exporting
`WANDB_API_KEY`:

```bash
export WANDB_API_KEY=your_key
# then set wandb.enable: true in your training YAML
./scripts/train.sh examples/train/pi05/cloud.yaml
```

## Checkpoints

Checkpoints are saved to `/scratch/outputs/<job_name>/` every `save_freq` steps.
The latest checkpoint is symlinked at `.../checkpoints/last`.

To push a checkpoint to HuggingFace Hub after training, set in your config:

```yaml
policy:
  push_to_hub: true
  repo_id: your-org/your-model-name
```
