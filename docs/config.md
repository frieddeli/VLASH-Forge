# Config Reference

Training is configured via YAML files in `examples/train/`. All values can be
overridden on the CLI by appending `key=value` arguments.

## Key configs

| File | Description |
|------|-------------|
| `examples/train/pi05/cloud.yaml` | π₀.₅ LoRA, cloud/HPC (recommended) |
| `examples/train/pi05/cloud_full.yaml` | π₀.₅ full fine-tuning via FSDP |
| `examples/train/pi05/async_lora.yaml` | π₀.₅ LoRA, local install |
| `examples/train/pi0/async_lora.yaml` | π₀ LoRA |

## Important fields

```yaml
policy:
  type: pi05                        # pi05 or pi0
  pretrained_path: lerobot/pi05_base  # HF Hub ID or local path
  push_to_hub: false                # set true to auto-upload checkpoint
  repo_id: your-org/your-model      # required when push_to_hub: true
  dtype: bfloat16
  gradient_checkpointing: false     # enable to save VRAM at ~20% speed cost

dataset:
  repo_id: ${DATASET_REPO_ID}       # HF Hub ID or absolute local path
  root: /scratch/.cache/lerobot     # local cache location
  video_backend: torchcodec         # torchcodec (GPU) or pyav (CPU/local)

output_dir: /scratch/outputs/pi05_cloud

batch_size: 1
grad_accum_steps: 8                 # effective batch = batch_size × NUM_GPUS × grad_accum_steps
steps: 50000
num_workers: 4

max_delay_steps: 8                  # temporal delay augmentation; 0 = sync training
shared_observation: true            # ~9× training speedup when max_delay_steps > 0

lora:
  enable: true                      # false = full fine-tuning (use TRAIN_BACKEND=fsdp)
  r: 16
  alpha: 16
  use_qlora: false                  # 4-bit quantised LoRA (fits in 8 GB)

wandb:
  enable: false                     # set true + export WANDB_API_KEY to enable
  project: vlash
```

## CLI overrides

```bash
# Override single values
./scripts/train.sh cloud.yaml batch_size=2 steps=10000

# Use a local dataset
./scripts/train.sh cloud.yaml \
  dataset.repo_id=/absolute/path/to/dataset \
  dataset.video_backend=pyav \
  output_dir=outputs/local_run
```
