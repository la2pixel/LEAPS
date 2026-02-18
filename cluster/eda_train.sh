#!/bin/bash
set -euo pipefail

source /etc/profile.d/modules.sh
module load cuda
module load cudnn

cd /lustre/fast/fast/lsivakumar/LEAPS
export PYTHONPATH="/lustre/fast/fast/lsivakumar/LEAPS/src:${PYTHONPATH:-}"

echo "=== GPU check ==="
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo "=== Starting EDA + Training ==="
python3 scripts/eda_and_train.py

echo "=== Done ==="
