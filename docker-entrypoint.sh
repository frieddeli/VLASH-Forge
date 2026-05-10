#!/bin/bash
set -e

# Normalize CUDA_VISIBLE_DEVICES from UUID to integer indices.
# PBSpro (NSCC ASPIRE) sets UUIDs like "GPU-50ee0fc4-bb3d-920c-8039-da7054e1496b".
# NCCL and DeepSpeed require integer indices (0,1,2,...).
# This block is a no-op on SLURM and commercial cloud where CVD is already integers.
if [[ "$CUDA_VISIBLE_DEVICES" == GPU-* ]] || [[ "$CUDA_VISIBLE_DEVICES" == *,GPU-* ]]; then
    N=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N - 1)))
    echo "[entrypoint] Normalized CUDA_VISIBLE_DEVICES → $CUDA_VISIBLE_DEVICES"
fi

# Default HF cache to /scratch so model weights persist across container restarts.
# /scratch is always a bind-mount from the host (EBS, HPC scratch, etc.) —
# see scripts/train.sh for how this is set up per environment.
export HF_HOME="${HF_HOME:-/scratch/.cache/huggingface}"

# Log into HuggingFace so gated models (pi0.5, pi0) can be downloaded.
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null
    echo "[entrypoint] HuggingFace login complete"
fi

exec pixi run accelerate launch \
    --config_file /workspace/deepspeed_config.yaml \
    --num_processes "${NUM_GPUS:-1}" \
    -m vlash.train "$@"
