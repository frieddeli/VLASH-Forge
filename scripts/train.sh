#!/bin/bash
# Unified VLASH training launcher for Docker (cloud/local) and Singularity (HPC).
#
# The container always sees /scratch as its persistent storage root:
#   /scratch/.cache/huggingface  — base model weights (HF_HOME)
#   /scratch/.cache/lerobot      — dataset cache
#   /scratch/outputs/            — checkpoints
#
# You bind whatever storage you have on your host to /scratch:
#   HPC (NSCC/SLURM):     SCRATCH=/scratch/users/ntu/your-id
#   AWS (EBS mounted):     SCRATCH=/mnt/ebs
#   GCP (Persistent Disk): SCRATCH=/mnt/pd
#   GCS FUSE:              SCRATCH=/mnt/gcs   (mount bucket first)
#   Local workstation:     SCRATCH=$HOME/vlash-scratch
#
# Usage:
#   export SCRATCH=/your/persistent/storage
#   export HF_TOKEN=hf_xxx
#   export DATASET_REPO_ID=your-org/your-dataset
#   ./scripts/train.sh examples/train/pi05/cloud.yaml
#
# Optional:
#   NUM_GPUS=4                  — number of GPUs (default: 1)
#   TRAIN_BACKEND=fsdp          — use FSDP for full fine-tuning (default: deepspeed for LoRA)
#   WANDB_API_KEY=xxx           — enable W&B logging (also set wandb.enable: true in config)

set -e

: ${SCRATCH:?"SCRATCH must be set — path to persistent storage on your host"}
: ${HF_TOKEN:?"HF_TOKEN must be set — HuggingFace token for gated model download"}

CONFIG=${1:-examples/train/pi05/cloud.yaml}

if [ -f "vlash.sif" ]; then
    echo "[train] Singularity — binding ${SCRATCH} → /scratch"
    singularity run --nv \
        -B "${SCRATCH}:/scratch" \
        --env HF_TOKEN="${HF_TOKEN}" \
        --env HF_HOME="/scratch/.cache/huggingface" \
        --env DATASET_REPO_ID="${DATASET_REPO_ID:-}" \
        --env NUM_GPUS="${NUM_GPUS:-1}" \
        --env TRAIN_BACKEND="${TRAIN_BACKEND:-deepspeed}" \
        --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
        vlash.sif "${CONFIG}"
else
    echo "[train] Docker — mounting ${SCRATCH} → /scratch"
    docker run --rm --gpus all \
        -v "${SCRATCH}:/scratch" \
        -e HF_TOKEN="${HF_TOKEN}" \
        -e HF_HOME="/scratch/.cache/huggingface" \
        -e DATASET_REPO_ID="${DATASET_REPO_ID:-}" \
        -e NUM_GPUS="${NUM_GPUS:-1}" \
        -e TRAIN_BACKEND="${TRAIN_BACKEND:-deepspeed}" \
        -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
        vlash:latest "${CONFIG}"
fi
