#!/bin/bash
# Full environment setup for LEAPS on the cluster.
# Run once from the repo root to create/refresh the venv.
#
# Usage:
#   cd /fast/lsivakumar/LEAPS
#   bash cluster/setup_env.sh

set -e

REPO_DIR="/fast/lsivakumar/LEAPS"
VENV_DIR="$REPO_DIR/.venv"
LOCO_MUJOCO_DIR="/home/lsivakumar/GitHub/Thesis/loco-mujoco"

echo "=== LEAPS Environment Setup ==="

# 1. Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
else
    echo "[1/4] Venv already exists at $VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

# 2. Install LEAPS + all pip-managed dependencies
echo "[2/4] Installing LEAPS (pip dependencies)"
pip install --upgrade pip --quiet
pip install -e "$REPO_DIR[dev,wandb]"

# 3. Install loco-mujoco from local source (NOT PyPI — PyPI version is incomplete)
echo "[3/4] Installing loco-mujoco from local source: $LOCO_MUJOCO_DIR"
pip install -e "$LOCO_MUJOCO_DIR"

# 4. Smoke test
echo "[4/4] Smoke test"
python3 -c "
import leaps
import loco_mujoco
import torch
print(f'  leaps        OK')
print(f'  loco_mujoco  OK  (version: {loco_mujoco.__version__})')
print(f'  torch        OK  (CUDA: {torch.cuda.is_available()})')
"

echo ""
echo "Done. Activate with: source $VENV_DIR/bin/activate"
