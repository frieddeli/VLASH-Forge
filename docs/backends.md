# Training Backends

Two backends are available, selected via the `TRAIN_BACKEND` environment variable.

| `TRAIN_BACKEND` | Method | Use case |
|----------------|--------|----------|
| `deepspeed` (default) | DeepSpeed ZeRO-2 | LoRA fine-tuning |
| `fsdp` | PyTorch FSDP FULL_SHARD | Full fine-tuning (`lora.enable: false`) |

## DeepSpeed ZeRO-2 (default)

Best for LoRA fine-tuning. Shards optimiser states and gradients across GPUs while
keeping model parameters replicated on each GPU, halving per-GPU memory vs. standard
DDP without the all-gather overhead of ZeRO-3.

- **Config:** `configs/deepspeed_config.yaml`
- **Mixed precision:** bf16 (A100/H100/RTX 40xx). Switch to `fp16` for V100.
- **Scale:** 1–N GPUs on a single node

!!! warning "V100 users"
    V100 GPUs do not support bf16 natively. PyTorch will silently emulate it in
    fp32, negating the memory benefit. Change `mixed_precision: fp16` in
    `configs/deepspeed_config.yaml` if running on V100 (AWS `p3` instances).

## FSDP FULL_SHARD

For full fine-tuning with `lora.enable: false`. Equivalent to DeepSpeed ZeRO-3 —
shards model parameters, gradients, and optimiser states across all GPUs, requiring
an all-gather before each forward pass.

- **Config:** `configs/fsdp_config.yaml`
- **Wrap policy:** `TRANSFORMER_BASED_WRAP` on `GemmaDecoderLayer`
- **Scale:** 1–N GPUs on a single node
- **Memory:** ~40 GB per GPU for π₀.₅, ~80 GB for π₀

```bash
TRAIN_BACKEND=fsdp ./scripts/train.sh examples/train/pi05/cloud_full.yaml
```

## Multi-node

Both backends are **single-node only**. Multi-node training (multiple machines)
requires per-node rank injection via SLURM `srun` or a Kubernetes operator and
is not currently supported.
