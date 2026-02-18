#!/bin/bash
# Training script for cluster GPU jobs.
# Called by train.sub — do not run directly on login nodes.

set -euo pipefail

# Load GPU modules
source /etc/profile.d/modules.sh
module load cuda
module load cudnn

# Activate the project venv
cd /lustre/fast/fast/lsivakumar/LEAPS
source /fast/lsivakumar/LEAPS/.venv/bin/activate

# Parse job type from first argument
JOB_TYPE="${1:-snapshot}"
shift || true

case "$JOB_TYPE" in
    snapshot)
        echo "=== Snapshot-level training ==="
        python3 -m leaps.scripts.train \
            --data /fast/lsivakumar/data/processed/emg_activations.h5 \
            --output-dir /fast/lsivakumar/LEAPS/experiments/snapshot_sweep \
            --wandb --wandb-project leaps \
            "$@"
        ;;
    stride)
        echo "=== Stride-level training ==="
        python3 -m leaps.scripts.train_strides \
            --data /fast/lsivakumar/data/processed/emg_activations.h5 \
            --output-dir /fast/lsivakumar/LEAPS/experiments/stride_sweep \
            --wandb --wandb-project leaps \
            "$@"
        ;;
    *)
        echo "Unknown job type: $JOB_TYPE"
        echo "Usage: train.sh [snapshot|stride] [extra args...]"
        exit 1
        ;;
esac

echo "=== Done ==="
