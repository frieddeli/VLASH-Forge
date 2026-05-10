# VLASH Training

Fine-tuning a state-of-the-art robot policy like π₀.₅ on your own task requires
multi-GPU distributed training infrastructure that has never been publicly released
for the VLASH framework. **This project removes that barrier.**

It packages [VLASH](https://arxiv.org/abs/2512.01031) — a VLA fine-tuning and
deployment framework built on [LeRobot](https://github.com/huggingface/lerobot)
— into a single container image that runs on a cloud VM, an HPC cluster, or a
local workstation with one command. You bring your robot demonstrations; the
pipeline handles the rest.

## Workflow

```
1. Collect demonstrations on your robot  →  upload to HuggingFace Hub
2. Run:  ./scripts/train.sh config.yaml  →  fine-tuned checkpoint on HF Hub
3. Load checkpoint on your inference hardware  →  deploy
```

## What VLASH adds over standard LeRobot

| Feature | Description |
|---------|-------------|
| Asynchronous inference | Temporal Delay Augmentation — 29.5× lower latency on embedded hardware |
| Memory-efficient fine-tuning | LoRA + shared observation encoding — fine-tune π₀.₅ on 12 GB VRAM |
| Distributed training | DeepSpeed ZeRO-2 (LoRA) and FSDP (full fine-tuning) |
| Portable containers | Same Docker/Singularity image on AWS, GCP, NSCC ASPIRE, or a local GPU |

## Validated end-to-end

The pipeline is validated on a ball pick-and-place task using a Piper arm with
inference on a Jetson AGX Orin at 30 Hz:

- **65% task success rate** with async inference vs. 5% synchronous baseline
- **29.5× latency reduction** — 184.5 ms vs. 5444.1 ms per inference call
- **2.9× faster task completion** — 52 s vs. 153 s

## Quick start

```bash
# 1. Set required environment variables
export SCRATCH=/your/persistent/storage   # mounted disk
export HF_TOKEN=hf_xxx                    # HuggingFace token
export DATASET_REPO_ID=your-org/your-dataset

# 2. Pull the container
singularity pull vlash.sif docker://frieddeli/vlash:latest
# or: docker pull frieddeli/vlash:latest

# 3. Train
./scripts/train.sh examples/train/pi05/cloud.yaml
```

See [First-time Setup](setup.md) for prerequisites.
