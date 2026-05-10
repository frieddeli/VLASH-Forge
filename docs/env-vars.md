# Environment Variables

All variables can be set in a `.env` file (copy from `.env.example`) or exported
in your shell before running `scripts/train.sh` or `docker-compose up`.

## Required

| Variable | Description |
|----------|-------------|
| `SCRATCH` | Path to persistent storage on the host, bind-mounted to `/scratch` inside the container. See [Storage Setup](storage.md). |
| `HF_TOKEN` | HuggingFace token with read+write access. Required to download gated models (π₀/π₀.₅) and push trained checkpoints. |
| `DATASET_REPO_ID` | HuggingFace dataset repo ID, e.g. `your-org/your-dataset`. Substituted into `dataset.repo_id` in the training config at runtime. |

## Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | `1` | Number of GPUs to use. Passed to `accelerate launch --num_processes`. |
| `TRAIN_BACKEND` | `deepspeed` | Training backend. `deepspeed` for LoRA, `fsdp` for full fine-tuning. |
| `WANDB_API_KEY` | — | W&B API key. Only needed when `wandb.enable: true` in your training config. |

## Inside the container

These are set automatically by the entrypoint and do not need to be set manually:

| Variable | Value | Description |
|----------|-------|-------------|
| `HF_HOME` | `/scratch/.cache/huggingface` | HuggingFace cache directory. Persists via the `/scratch` bind-mount. |

## Example `.env`

```bash
# Copy .env.example to .env and fill in:
HF_TOKEN=hf_xxx
DATASET_REPO_ID=your-org/your-dataset
SCRATCH=/your/persistent/storage
NUM_GPUS=4
TRAIN_BACKEND=deepspeed
WANDB_API_KEY=
```
