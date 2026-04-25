# Agent Guidelines — vlash (Comp-4901D)

## Repository identity
- **Local path**: `/home/users/ntu/m230060/vlash`
- **Remote**: `https://github.com/frieddeli/Comp-4901D` (branch: `main`)
- This is the **primary / upstream-fork** repo. Do NOT confuse it with `vlash_lora`.

## Working directory
Always confirm you are inside `/home/users/ntu/m230060/vlash` before running any
command or editing any file. If in doubt:

```bash
cd /home/users/ntu/m230060/vlash
```

## Pixi environment
All Python commands must be run through the **pixi** managed environment located at:

```
/home/users/ntu/m230060/vlash/.pixi/envs/default/
```

Activate or prefix commands correctly:

```bash
# Preferred — run a one-off command inside the env
pixi run <command>

# Or invoke the interpreter directly
/home/users/ntu/m230060/vlash/.pixi/envs/default/bin/python <script>
```

- Python version: **3.12**
- Platform: `linux-64`
- Channels: `conda-forge`
- The package is installed in editable mode (`pip install -e .`), so source edits take effect immediately without reinstalling.
- Do **not** use `conda`, `venv`, or system Python for this repo.

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

## Scratch storage
Large outputs (checkpoints, datasets, logs) live on the scratch filesystem, **not** in the repo directory:

```
/scratch/users/ntu/m230060/
├── outputs/train/      # training checkpoints & logs
└── comp4901/           # datasets
```

Always write checkpoints and heavy artifacts to `/scratch/users/ntu/m230060/` — the home directory (`/home/users/ntu/m230060/`) has limited quota and should only contain source code.

## General rules
1. Never install packages globally; only modify `pyproject.toml` to add dependencies so pixi can manage them.
2. Do not commit generated files (`.pixi/`, `__pycache__/`, `*.egg-info/`, `outputs/`).
3. Run tests with `pixi run python -m pytest` from the repo root before marking a task done.
4. If you are unsure whether a change belongs in `vlash` or `vlash_lora`, check the remote URL — this repo points to **frieddeli/Comp-4901D**.
