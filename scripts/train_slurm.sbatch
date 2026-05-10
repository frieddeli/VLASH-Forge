#!/bin/bash
# SLURM job script for VLASH training.
# Tested on SLURM-based HPC clusters (e.g. AWS HPC, most academic clusters).
# For NSCC ASPIRE (PBS/PBSpro), use a .pbs script instead.
#
# Usage:
#   export SCRATCH=/scratch/your-id  HF_TOKEN=hf_xxx  DATASET_REPO_ID=org/ds
#   sbatch scripts/train_slurm.sbatch examples/train/pi05/cloud.yaml
#
# Or with a .env file:
#   set -a && source .env && set +a
#   sbatch scripts/train_slurm.sbatch examples/train/pi05/cloud.yaml

#SBATCH --job-name=vlash-train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                # number of GPUs — match NUM_GPUS below
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
# Uncomment and set your cluster's GPU partition name if needed:
# #SBATCH --partition=gpu

set -e

: ${SCRATCH:?"SCRATCH must be set — path to persistent storage on this cluster"}
: ${HF_TOKEN:?"HF_TOKEN must be set"}

CONFIG=${1:-examples/train/pi05/cloud.yaml}

# Load Singularity/Apptainer module — name varies by cluster
# Common names: singularity, apptainer, singularity-ce
module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true

mkdir -p logs

echo "[slurm] Job ${SLURM_JOB_ID} — config: ${CONFIG}"
echo "[slurm] SCRATCH=${SCRATCH}, NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}"

singularity run --nv \
    -B "${SCRATCH}:/scratch" \
    --env HF_TOKEN="${HF_TOKEN}" \
    --env HF_HOME="/scratch/.cache/huggingface" \
    --env DATASET_REPO_ID="${DATASET_REPO_ID:-}" \
    --env NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}" \
    --env TRAIN_BACKEND="${TRAIN_BACKEND:-deepspeed}" \
    --env WANDB_API_KEY="${WANDB_API_KEY:-}" \
    vlash.sif "${CONFIG}"
