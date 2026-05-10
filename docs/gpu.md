# GPU Requirements

## Memory by model and mode

| Model | Mode | Min VRAM per GPU | Recommended |
|-------|------|-----------------|-------------|
| π₀.₅ (1.3B) | LoRA (`deepspeed`) | 12 GB | RTX 3090 / A10G / T4 |
| π₀.₅ (1.3B) | Full (`fsdp`) | 40 GB | 4×A100 40GB |
| π₀ (3B) | LoRA (`deepspeed`) | 24 GB | RTX 4090 / A100 40GB |
| π₀ (3B) | Full (`fsdp`) | 80 GB | 4×A100 80GB / 4×H100 |

## AWS instance guide

| Use case | Instance | GPUs |
|----------|----------|------|
| π₀.₅ LoRA, single GPU | `g5.xlarge` | 1×A10G 24GB |
| π₀.₅ LoRA, multi-GPU | `g5.12xlarge` | 4×A10G 24GB |
| π₀.₅ full fine-tuning | `p4d.24xlarge` | 8×A100 40GB |
| π₀ full fine-tuning | `p4de.24xlarge` | 8×A100 80GB |

## Out of memory (OOM)

Apply these fixes in order until training fits:

1. **Reduce `batch_size` to 1** in your training config
2. **Increase `grad_accum_steps`** to maintain the same effective batch size
3. **Enable `gradient_checkpointing: true`** in the policy section
   (~20% slower, saves ~30% VRAM)
4. **Reduce `lora.r` from 16 to 8** — halves LoRA parameter count
5. **Switch to QLoRA** — set `lora.use_qlora: true`
   (4-bit base model, fits π₀.₅ in 8 GB)
6. **Full fine-tuning only** — enable `fsdp_activation_checkpointing: true`
   in `configs/fsdp_config.yaml`
