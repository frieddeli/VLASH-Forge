# Running Training

## Prerequisites

- `SCRATCH`, `HF_TOKEN`, and `DATASET_REPO_ID` set in your environment
  (see [Storage Setup](storage.md) and [First-time Setup](setup.md))
- Container image available (Docker or Singularity)

```bash
# Docker (cloud / local)
docker pull frieddeli/vlash-forge:latest
# or build locally:
docker build -t vlash-forge:latest .

# Singularity (HPC)
singularity pull vlash.sif docker://frieddeli/vlash-forge:latest
```

---

## Option A — Unified launcher (Docker or Singularity)

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

# Full fine-tuning (40 GB+ VRAM per GPU)
TRAIN_BACKEND=fsdp ./scripts/train.sh examples/train/pi05/cloud_full.yaml
```

!!! tip "Smoke test before a full run"
    Validate the pipeline works before committing to 50k steps:
    ```bash
    ./scripts/train.sh examples/train/pi05/cloud.yaml steps=100 save_freq=100 log_freq=10
    ```
    This confirms the dataset loads, model downloads, and a checkpoint saves — without using significant compute quota.

!!! info "First-run overhead"
    DeepSpeed and bitsandbytes compile CUDA extensions on first use (~1–3 minutes).
    This is cached in `$SCRATCH/.cache` and does not repeat on subsequent runs.

---

## Option B — Docker Compose (cloud, multi-GPU)

For commercial cloud VMs, SSH into the instance and run directly — no scheduler needed.

```bash
cp .env.example .env   # fill in HF_TOKEN, DATASET_REPO_ID, SCRATCH, NUM_GPUS
docker-compose up
```

The difference from HPC: on a cloud VM you are already on the GPU node, so there
is no job submission step. Set your env vars, run the command, and training starts immediately.

---

## Option C — PBS job (NSCC ASPIRE and PBS clusters)

NSCC ASPIRE uses PBSpro. Create a job script and submit with `qsub`:

```bash
#!/bin/bash
#PBS -l select=1:ngpus=4
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o logs/

export SCRATCH=/scratch/users/ntu/<your-id>
export HF_TOKEN=hf_xxx
export DATASET_REPO_ID=your-org/your-dataset
export WANDB_API_KEY=your_wandb_key   # optional — remove if not using W&B

cd $PBS_O_WORKDIR
./scripts/train.sh examples/train/pi05/cloud.yaml
```

```bash
mkdir -p logs
qsub scripts/train_pbs.pbs
qstat -u $USER   # check job status
```

!!! info "CUDA_VISIBLE_DEVICES on PBS"
    PBSpro sets GPU IDs as UUIDs (e.g. `GPU-50ee0fc4-...`). The container entrypoint
    automatically normalises these to integer indices (`0,1,2,...`) required by NCCL.
    No manual export needed.

---

## Option D — SLURM job

```bash
export SCRATCH=/scratch/users/ntu/<your-id>
export HF_TOKEN=hf_xxx
export DATASET_REPO_ID=your-org/your-dataset

sbatch scripts/train_slurm.sbatch examples/train/pi05/cloud.yaml
```

Edit the `#SBATCH` directives at the top of `scripts/train_slurm.sbatch` to match
your cluster's partition name, GPU count, and time limit.

---

## Option E — Kubernetes

```bash
kubectl create secret generic hf-secret --from-literal=token=<YOUR_HF_TOKEN>
kubectl apply -f k8s/training-job.yaml
kubectl logs -f job/vlash-train
```

---

## W&B logging

`wandb` is **pre-installed in the container** — no additional dependencies needed.
To enable it:

1. Set `wandb.enable: true` in your training config
2. Add your API key to the job script or `.env`:

```bash
export WANDB_API_KEY=your_key   # PBS / SLURM job script
# or in .env for Docker Compose:
WANDB_API_KEY=your_key
```

W&B streams loss, grad norm, and lr every `log_freq` steps. If a run crashes
mid-way, the dashboard shows exactly which step it reached — the most useful
tool for diagnosing training failures.

Get your API key at [wandb.ai/settings](https://wandb.ai/settings).

---

## Debugging a failed run

| Symptom | Where to look |
|---------|--------------|
| Job never starts | PBS: `qstat -u $USER` / SLURM: `squeue -u $USER` — check queue state |
| Job starts then exits immediately | `logs/` job output file — look for missing env vars or container errors |
| `CUDA error` or `OOM` | Job log + see [GPU Requirements → OOM](gpu.md#out-of-memory-oom) |
| Training starts but loss is NaN | Enable W&B and check the loss curve — usually lr too high or bad data |
| Checkpoint not saved | Check `$SCRATCH/outputs/` exists and has write permission |
| Model weights not downloading | Verify `HF_TOKEN` is set and model license is accepted (see [Setup](setup.md#2-accept-the-base-model-licenses)) |

---

## Checkpoints

Checkpoints are saved to `/scratch/outputs/<job_name>/checkpoints/` every `save_freq` steps.
The latest is symlinked at `.../checkpoints/last`.

To push to HuggingFace Hub automatically after training:

```yaml
policy:
  push_to_hub: true
  repo_id: your-org/your-model-name
```
