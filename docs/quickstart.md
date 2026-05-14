# Quickstart

Get from zero to a running training job in 5 steps.
For detailed explanations of each step, see [Getting Started](setup.md).

---

## How the online services connect

```
HuggingFace Hub                        Weights & Biases
  lerobot/pi05_base ──► pulls at start      (optional)
  your-org/dataset  ──► pulls at start          ▲
  your-org/model    ◄── pushes on finish         │ live metrics
                                                  │
                              Training run (your GPU)
```

`HF_TOKEN` authenticates all Hub transfers. `WANDB_API_KEY` + `wandb.enable: true`
activates live metric streaming. Nothing is baked into the container — all
credentials are passed as environment variables at runtime.

---

## Step 1 — One-time setup

Accept the model licenses on HuggingFace *(click Agree and access repository)*:

- [huggingface.co/lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)
- [huggingface.co/lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base)

Upload your dataset:

```bash
huggingface-cli login --token $HF_TOKEN
huggingface-cli repo create your-dataset-name --type dataset --private
huggingface-cli upload your-hf-username/your-dataset-name /path/to/dataset/ --repo-type dataset
```

## Step 2 — Pull the container

=== "HPC (Singularity)"
    ```bash
    singularity pull vlash.sif docker://frieddeli/vlash-forge:latest
    ```

=== "Cloud / Local (Docker)"
    ```bash
    docker pull frieddeli/vlash-forge:latest
    ```

## Step 3 — Set environment variables

```bash
export SCRATCH=/your/persistent/storage   # see Storage Setup for options
export HF_TOKEN=hf_xxx
export DATASET_REPO_ID=your-hf-username/your-dataset-name
```

## Step 4 — Smoke test

```bash
./scripts/train.sh examples/train/pi05/cloud.yaml steps=100 save_freq=100 log_freq=10
```

Confirms dataset loads, model downloads, and a checkpoint saves. If this passes, proceed.

## Step 5 — Full run

=== "HPC (PBS)"
    ```bash
    qsub scripts/train_pbs.pbs
    tail -f logs/<job-id>.o
    ```

=== "HPC (SLURM)"
    ```bash
    sbatch scripts/train_slurm.sbatch examples/train/pi05/cloud.yaml
    ```

=== "Cloud / Local (Docker)"
    ```bash
    cp .env.example .env   # fill in HF_TOKEN, DATASET_REPO_ID, SCRATCH
    docker-compose up
    ```

Checkpoint lands at `$SCRATCH/outputs/pi05_cloud/checkpoints/last/pretrained_model/`.
Set `push_to_hub: true` in `cloud.yaml` to upload it to HF Hub automatically.

---

**Need more detail?** See [First-time Setup](setup.md), [Storage Setup](storage.md),
or [Running Training](running.md).
