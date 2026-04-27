# Agent Guidelines — vlash (Comp-4651)

## Repository identity

**We are adapting a fine tuning pipeline (VLASH) originally built for HPC clusters and attempting to dockerise and simplfy the setup so that it can be deployed on commercial cloud instances - we have validated the code works on the Singapore NCSS ASPIRE 2A cluter with PBS.**

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

## Below are NSCC specific, use only as reference

## Scratch storage

Large outputs (checkpoints, datasets, logs) live on the scratch filesystem, **not** in the repo directory:

```
/scratch/users/ntu/m230060/
├── outputs/train/      # training checkpoints & logs
└── comp4901/           # datasets
```

Always write checkpoints and heavy artifacts to `/scratch/users/ntu/m230060/` — the home directory (`/home/users/ntu/m230060/`) has limited quota and should only contain source code.
