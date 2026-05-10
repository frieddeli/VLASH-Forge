# First-time Setup

Before running training you need three things: a HuggingFace account with a token,
access to the gated base models, and your dataset uploaded to HF Hub.
This is a one-time process per account.

## 1. HuggingFace token

Create an account at [huggingface.co](https://huggingface.co) if you don't have one.
Generate a token with **write** permission at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) —
write access is needed to push trained checkpoints back to the Hub.

```bash
export HF_TOKEN=hf_xxx   # add to your shell profile or .env
```

!!! warning
    Never commit `HF_TOKEN` to the repo. Add `.env` to your `.gitignore`.

## 2. Accept the base model licenses

VLASH fine-tunes PaliGemma-based models that are gated. You must accept the
license on the HF website once before your token can download the weights.

Visit each model page and click **Agree and access repository**:

- **π₀.₅** — [huggingface.co/lerobot/pi05_base](https://huggingface.co/lerobot/pi05_base)
- **π₀** — [huggingface.co/lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base)

Acceptance is per-account and propagates immediately. The container downloads
weights automatically on first run and caches them in `$SCRATCH/.cache/huggingface`.

## 3. Upload your dataset

Your dataset must be in [LeRobot format](https://github.com/huggingface/lerobot)
(`data/`, `videos/`, `meta/` folders with a `meta/info.json`).

```bash
# Authenticate
huggingface-cli login --token $HF_TOKEN

# Create the repo
huggingface-cli repo create your-dataset-name --type dataset --private

# Upload — preserves directory structure exactly
huggingface-cli upload your-hf-username/your-dataset-name \
  /path/to/local/lerobot/dataset/ \
  --repo-type dataset
```

Set `DATASET_REPO_ID=your-hf-username/your-dataset-name` when running training.

!!! tip "Team datasets"
    Create a [HuggingFace organisation](https://huggingface.co/organizations/new),
    push under `your-org/your-dataset-name`, and invite collaborators at
    `huggingface.co/your-org` → Settings → Members.

## 4. Push trained checkpoints (optional)

Set `push_to_hub: true` and `repo_id: your-org/your-model` in your training config
to automatically upload the final checkpoint after training. Recommended for HPC
clusters where scratch storage is purged periodically.
