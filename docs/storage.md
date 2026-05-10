# Storage Setup

Every environment needs a persistent directory that the container mounts as `/scratch`.
Model weights, datasets, and checkpoints all live here — nothing important is written
inside the container itself.

## The `/scratch` layout

```
$SCRATCH/
  .cache/huggingface/   ← base model weights (~10 GB, downloaded once)
  .cache/lerobot/       ← dataset videos and parquet files
  outputs/              ← training checkpoints
```

## Set `SCRATCH` for your environment

=== "HPC (NSCC ASPIRE / SLURM)"

    ```bash
    export SCRATCH=/scratch/users/ntu/<your-id>
    ```

=== "AWS (EBS volume)"

    ```bash
    # Mount EBS first, then:
    export SCRATCH=/mnt/ebs
    ```

=== "GCP (Persistent Disk)"

    ```bash
    # Mount PD first, then:
    export SCRATCH=/mnt/pd
    ```

=== "GCP (GCS FUSE)"

    ```bash
    # Mount bucket first: gcsfuse your-bucket /mnt/gcs
    export SCRATCH=/mnt/gcs
    ```

=== "Local workstation"

    ```bash
    export SCRATCH=$HOME/vlash-scratch
    mkdir -p $SCRATCH
    ```

## How it works

The container always sees `/scratch`. The launcher (`scripts/train.sh`) binds
your `$SCRATCH` path to `/scratch` inside the container — whether via
`singularity run -B` or `docker run -v`. The training config hardcodes
`/scratch`-relative paths so it works identically in every environment.

This means:

- **HPC parallel filesystem** (Lustre/GPFS), **cloud block storage** (EBS, PD),
  and **FUSE-mounted object storage** (GCS, S3) all work the same way.
- Model weights are cached on first download and reused across runs as long as
  `$SCRATCH` points to persistent storage.
- Checkpoints survive container exit because they are written to the host filesystem.

!!! note "Kubernetes"
    For Kubernetes, PersistentVolumeClaims in `k8s/training-job.yaml` replace the
    bind-mount. The underlying storage (EBS, Filestore, GCS) is configured in the
    PVC spec — the container still reads `/scratch` via a `volumeMount`.
