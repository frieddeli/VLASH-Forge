# Quickstart

This page walks through the complete flow from zero to a trained checkpoint,
covering both HPC (NSCC ASPIRE / PBS) and commercial cloud (Docker).

## How the online services connect

Three external services are involved in every training run:

```
┌─────────────────────────────────────────────────────────────────┐
│                      HuggingFace Hub                            │
│                                                                 │
│  lerobot/pi05_base ──── pulled at run start (cached to SCRATCH) │
│  your-org/your-dataset ─ pulled at run start (cached to SCRATCH)│
│  your-org/your-model ─── pushed when training completes         │
└────────────────────────────┬────────────────────────────────────┘
                             │  HF_TOKEN authenticates all three
                             ▼
                    ┌─────────────────┐
                    │  Training run   │  ← your GPU (HPC or cloud)
                    │  (container)    │
                    └────────┬────────┘
                             │  WANDB_API_KEY
                             ▼
                    ┌─────────────────┐
                    │  Weights &      │
                    │  Biases (W&B)   │  ← live loss, grad norm, lr
                    └─────────────────┘
```

**HuggingFace Hub** serves as the single source of truth for all models and
datasets. The container pulls the base model weights and your dataset
automatically at training start using `HF_TOKEN`, caches them to `$SCRATCH`,
and pushes the final checkpoint back when training completes.

**Weights & Biases** receives live training metrics (loss, grad norm, lr)
streamed every `log_freq` steps. It is optional but strongly recommended —
if a run crashes you can see exactly which step it reached and what the loss
curve looked like, without waiting for the job to finish.

**Nothing is baked into the container.** All credentials, dataset IDs, and
model paths are passed as environment variables at runtime, so the same image
works for any task or user.

---

---

## Step 1 — HuggingFace account and token

Create an account at [huggingface.co](https://huggingface.co) and generate a
token with **write** access at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Keep it somewhere safe — you will pass it as `HF_TOKEN` in every run.

---

## Step 2 — Accept the model licenses

VLASH fine-tunes gated models. Visit each page and click
**Agree and access repository** once:

- [huggingface.co/lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base) — π₀.₅ (recommended, 1.3B)
- [huggingface.co/lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base) — π₀ (3B)

The container downloads the weights automatically on first run and caches them.
You never need to do this again for the same HF account.

---

## Step 3 — Upload your dataset

Your dataset must be in [LeRobot format](https://github.com/huggingface/lerobot)
(`data/`, `videos/`, `meta/` with a `meta/info.json`).

```bash
huggingface-cli login --token $HF_TOKEN

huggingface-cli repo create your-dataset-name --type dataset --private

huggingface-cli upload your-hf-username/your-dataset-name \
  /path/to/local/dataset/ \
  --repo-type dataset
```

Note your dataset repo ID — you will use it as `DATASET_REPO_ID`.

---

## Step 4 — Pull the container

=== "HPC (Singularity)"

    Run this on the HPC login node:

    ```bash
    singularity pull vlash-forge.sif docker://frieddeli/vlash-forge:latest
    # ~10 GB download, takes 5–10 minutes
    # copy to scratch for faster job start:
    cp vlash-forge.sif $SCRATCH/vlash-forge.sif
    ```

=== "Cloud / Local (Docker)"

    ```bash
    docker pull frieddeli/vlash-forge:latest
    # or build from source:
    docker build -t vlash-forge:latest .
    ```

---

## Step 5 — Set up persistent storage

Every run needs a `SCRATCH` directory that the container mounts as `/scratch`.
This is where model weights, dataset cache, and checkpoints all live.

```bash
# HPC
export SCRATCH=/scratch/users/ntu/<your-id>

# AWS EBS / GCP Persistent Disk
export SCRATCH=/mnt/storage

# Local workstation
export SCRATCH=$HOME/vlash-scratch && mkdir -p $SCRATCH
```

---

## Step 6 — Smoke test (100 steps)

Before submitting a full run, verify the pipeline works end-to-end:

=== "HPC (PBS)"

    ```bash
    export HF_TOKEN=hf_xxx
    export DATASET_REPO_ID=your-hf-username/your-dataset-name

    # run interactively on a login node (CPU only, just tests imports + data loading)
    singularity run \
        -B ${SCRATCH}:/scratch \
        --env HF_TOKEN="${HF_TOKEN}" \
        --env HF_HOME="/scratch/.cache/huggingface" \
        --env DATASET_REPO_ID="${DATASET_REPO_ID}" \
        vlash-forge.sif examples/train/pi05/cloud.yaml \
        steps=100 save_freq=100 log_freq=10
    ```

=== "Cloud / Local (Docker)"

    ```bash
    export SCRATCH=$HOME/vlash-scratch
    export HF_TOKEN=hf_xxx
    export DATASET_REPO_ID=your-hf-username/your-dataset-name

    ./scripts/train.sh examples/train/pi05/cloud.yaml \
      steps=100 save_freq=100 log_freq=10
    ```

This confirms:

- [x] Dataset downloads and loads correctly
- [x] Model weights download from HF Hub
- [x] Training loop starts without error
- [x] A checkpoint is written to `$SCRATCH/outputs/`

If this passes, proceed to the full run.

---

## Step 7 — Full training run

=== "HPC (PBS)"

    Create a job script (or use `scripts/train_pbs.pbs`):

    ```bash
    #!/bin/bash
    #PBS -l select=1:ngpus=4:ncpus=16:mem=64gb
    #PBS -l walltime=24:00:00
    #PBS -j oe
    #PBS -o logs/
    #PBS -N vlash-forge

    export SCRATCH=/scratch/users/ntu/<your-id>
    export HF_TOKEN=hf_xxx
    export DATASET_REPO_ID=your-hf-username/your-dataset-name
    export WANDB_API_KEY=your_wandb_key   # optional

    cd $PBS_O_WORKDIR
    ./scripts/train.sh examples/train/pi05/cloud.yaml
    ```

    Submit:
    ```bash
    mkdir -p logs
    qsub scripts/train_pbs.pbs

    # monitor
    qstat -u $USER
    tail -f logs/<job-id>.o
    ```

=== "Cloud / Local (Docker)"

    ```bash
    export SCRATCH=/mnt/storage
    export HF_TOKEN=hf_xxx
    export DATASET_REPO_ID=your-hf-username/your-dataset-name
    export WANDB_API_KEY=your_wandb_key   # optional
    export NUM_GPUS=4

    ./scripts/train.sh examples/train/pi05/cloud.yaml

    # or with Docker Compose:
    cp .env.example .env   # fill in values
    docker-compose up
    ```

!!! info "Training time"
    50k steps on 4×A100 with π₀.₅ LoRA takes approximately 4–6 hours.
    Checkpoints are saved every `save_freq` steps (default: 10,000).

!!! info "W&B"
    `wandb` is pre-installed in the container. Set `wandb.enable: true` in
    `cloud.yaml` and export `WANDB_API_KEY` to stream live metrics. Useful
    for catching divergence or OOM without waiting for the job to finish.

---

## Step 8 — Find your checkpoint

After training completes, checkpoints are at:

```
$SCRATCH/outputs/pi05_cloud/
  checkpoints/
    000010000/   ← step 10k
    000020000/   ← step 20k
    ...
    last/        ← symlink to latest checkpoint
      pretrained_model/   ← inference-ready merged weights
```

The `pretrained_model/` folder inside each checkpoint is a standard HuggingFace
model directory — load it directly with LeRobot or VLASH inference.

---

## Step 9 — Push checkpoint to HuggingFace Hub

=== "Automatic (recommended)"

    Set these in your training config before the run:

    ```yaml
    policy:
      push_to_hub: true
      repo_id: your-hf-username/your-model-name
      private: true
    ```

    The final checkpoint uploads automatically when training completes.

=== "Manual"

    ```bash
    huggingface-cli upload your-hf-username/your-model-name \
      $SCRATCH/outputs/pi05_cloud/checkpoints/last/pretrained_model/ \
      --repo-type model
    ```

---

## What's next

- Load the checkpoint for inference with `vlash run examples/inference/async.yaml`
- Adjust hyperparameters in `examples/train/pi05/cloud.yaml` for your task
- See [GPU Requirements](gpu.md) if you hit OOM
- See [Config Reference](config.md) for all available options
