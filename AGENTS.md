# Agent Guidelines — vlash (Comp-4651)

## Repository identity

**We are adapting a fine tuning pipeline (VLASH) originally built for HPC clusters and attempting to dockerise and simplfy the setup so that it can be deployed on commercial cloud instances - we have validated the code works on the Singapore NCSS ASPIRE 2A cluter with PBS.**

VLASH paper can be found @ https://arxiv.org/html/2512.01031v1

## The Problem

VLASH team used DPP distributed training on 4xH100 but their repo did not included the DPP code, Distributed training is hard , fine tuning VLAs is hard

## Solution

 we attempt to implement distributed training using deepspeed

## Supported Models

VLASH currently supports **pi0** and **pi05** only — hardcoded in `vlash/policies/factory.py`. Both are PaliGemma-backbone transformer VLAs.

**SmolVLA** (HuggingFace's lightweight VLA, same transformer family) is a strong candidate for future extension — it shares the same attention/MLP structure so LoRA target modules and the training pipeline would carry over with a new `vlash/policies/smolvla/` directory and a factory entry. Do not attempt to add SmolVLA support without first verifying its action-chunking output format matches pi0/pi05's interface (the async inference overlap and `forward_shared_observation` depend on this).

ACT, Diffusion Policy, and TDMPC are **not** compatible with VLASH's async inference or shared observation optimisations and would require major rework.

## Guiding Implementation Document

**`IMPLEMENTATION_PLAN.md` at the repo root is the authoritative guide for this project.** Read it before starting any implementation work. It covers:

- All confirmed design decisions (base image, pixi strategy, HF model download approach, GPU targets)
- Portability audit findings and fixes
- Full file-by-file implementation plan across 5 phases
- Verification steps

## Assignment Grounding

Refer to the assignment.txt for info on what are the deliverables for this project - see submission requirements

## Report format

The project report should be written in QMD(quarto) as per the guidelines laid out by assignment.txt in assigment grounding rules

## Deliverables

1. Docker Images that can be deployed on commerical cloud providers or HPC enviroments through singularity containers https://docs.sylabs.io/guides/3.5/user-guide/introduction.html, https://docs.sylabs.io/guides/3.5/user-guide/oci_runtime.html
2. Report in QMD
3. Easy configuartion / setup to allow non technical user to avoid having to debug distrbuted training

## Pixi environment

All Python commands must be run through the **pixi** managed environment

Activate or prefix commands correctly:

```bash
# Preferred — run a one-off command inside the env
pixi run <command>

```

- Python version: **3.12**
- Platform: `linux-64`
- Channels: `conda-forge`
- The package is installed in editable mode (`pip install -e .`), so source edits take effect immediately without reinstalling.
- Do **not** use `conda`, `venv`, or system Python for this repo.

## General rules

1. Never install packages globally; only modify `pyproject.toml` to add dependencies so pixi can manage them.
2. Do not commit generated files (`.pixi/`, `__pycache__/`, `*.egg-info/`, `outputs/`).
3. Run tests with `pixi run python -m pytest` from the repo root before marking a task done.

## Change log

Every agent session that modifies files in this repo must append an entry to
`CHANGELOG.md` (create it at the repo root if it does not exist yet) using the
format below. Add the entry **before committing** any changes.

```markdown
## [YYYY-MM-DD] <short summary>

### Changed
- `path/to/file.py`: description of what was changed and why

### Added / Removed
- …
```

Keep entries concise — one bullet per file changed is sufficient.

## NSCC ASPIRE HPC Runbook

Complete steps to pull the project and run training on NSCC ASPIRE 2A.

### 1. First-time setup (login node)

```bash
# Set your scratch path
export SCRATCH=/scratch/users/ntu/<your-id>
mkdir -p $SCRATCH/outputs $SCRATCH/logs

# Clone the repo
cd $SCRATCH
git clone https://github.com/frieddeli/vlash-forge.git
cd vlash-forge

# Pull the Singularity image (~10 GB, takes 5–10 min)
singularity pull $SCRATCH/vlash-forge.sif docker://frieddeli/vlash-forge:latest
```

### 2. Set environment variables

```bash
export SCRATCH=/scratch/users/ntu/<your-id>
export HF_TOKEN=hf_xxx                           # HF token with read access to lerobot/pi05_base
export DATASET_REPO_ID=your-org/your-dataset     # HF dataset repo ID
```

### 3. Smoke test (login node, no GPU)

Confirms dataset loads and model downloads before burning quota:

```bash
singularity run \
  -B ${SCRATCH}:/scratch \
  --env HF_TOKEN="${HF_TOKEN}" \
  --env HF_HOME="/scratch/.cache/huggingface" \
  --env DATASET_REPO_ID="${DATASET_REPO_ID}" \
  ${SCRATCH}/vlash-forge.sif \
  examples/train/pi05/cloud.yaml steps=100 save_freq=100 log_freq=10
```

### 4. Full training job (PBS)

Edit `scripts/train_pbs.pbs` to set your SCRATCH, HF_TOKEN, and DATASET_REPO_ID at the top, then:

```bash
mkdir -p logs
qsub scripts/train_pbs.pbs                    # async + LoRA (default cloud.yaml)

# Monitor
qstat -u $USER
tail -f logs/<job-id>.o
```

Checkpoint lands at: `$SCRATCH/outputs/pi05_cloud/checkpoints/last/`

### 5. Sync baseline job (for comparison)

```bash
cp examples/train/pi05/cloud.yaml examples/train/pi05/cloud_sync.yaml
# Edit cloud_sync.yaml: max_delay_steps: 0, shared_observation: false,
#   output_dir: /scratch/outputs/pi05_sync, job_name: pi05_sync

qsub scripts/train_pbs.pbs examples/train/pi05/cloud_sync.yaml
```

### 6. Scaling experiment (1 / 2 / 4 GPU — measures ZeRO-2 throughput)

Run 500 steps at each GPU count, compare `update_s` in logs:

```bash
sed 's/ngpus=4/ngpus=1/' scripts/train_pbs.pbs | \
  PBS_EXTRA="output_dir=/scratch/outputs/scale_1gpu job_name=scale_1gpu steps=500" qsub

sed 's/ngpus=4/ngpus=2/' scripts/train_pbs.pbs | \
  PBS_EXTRA="output_dir=/scratch/outputs/scale_2gpu job_name=scale_2gpu steps=500" qsub

qsub scripts/train_pbs.pbs \
  examples/train/pi05/cloud.yaml \
  output_dir=/scratch/outputs/scale_4gpu job_name=scale_4gpu steps=500
```

Throughput = effective_batch_size / update_s. Near-linear scaling validates ZeRO-2.

### 7. Push checkpoint to HF Hub after training

```bash
huggingface-cli upload your-org/vlash-models \
  $SCRATCH/outputs/pi05_cloud/checkpoints/last/pretrained_model/ \
  --repo-type model
```

Or set `push_to_hub: true` + `repo_id: your-org/vlash-models` in `cloud.yaml` before submitting.

### Scratch storage layout

```
$SCRATCH/
  vlash-forge.sif              ← Singularity image
  vlash-forge/                 ← repo clone
  .cache/huggingface/          ← base model weights (~10 GB, cached after first run)
  .cache/lerobot/              ← dataset cache
  outputs/pi05_cloud/          ← async+LoRA checkpoints
  outputs/pi05_sync/           ← sync baseline checkpoints
  outputs/scale_*/             ← scaling experiment checkpoints
  logs/                        ← PBS job stdout/stderr
```

Home directory (`/home/users/ntu/<your-id>/`) has limited quota — keep only source code there.
