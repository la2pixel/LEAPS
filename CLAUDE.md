# LEAPS — Project Guide

> EMG-based latent action priors for muscle-actuated humanoid locomotion.
> Author: Lalitha Sivakumar, MPI IS Tübingen. Target: ICLR.

@.claude/Architecture.md
@.claude/Status.md
@.claude/Papers.md

---

## Tech Stack
- Python 3.10 | PyTorch (primary), JAX/Brax (secondary)
- MuJoCo, Gymnasium, Stable-Baselines3, LocoMuJoCo
  - LocoMuJoCo editable install: `/home/lsivakumar/GitHub/Thesis/loco-mujoco/`
- NumPy, SciPy, pandas, h5py, scikit-learn
- Hydra + OmegaConf | TensorBoard, W&B
- Build: setuptools via pyproject.toml

## Structure
```
src/leaps/
├── data/        # EMG preprocessing + HDF5 loading + PyTorch datasets
├── models/      # AE/VAE/WAE/NMF/PCA models
├── training/    # Training loops
├── evaluation/  # Gait-aware metrics
├── envs/        # LocoEnv wrapper + EMG mapping + rewards + terminal state
└── scripts/     # CLI entry points only
```
Data (gitignored): `data/`, `experiments/`, `checkpoints/`, `logs/`, `wandb/`, `renders/`

## Commands
```bash
# Setup
pip install -e ".[dev]"

# Quality
black src/ tests/ --line-length 100
isort src/ tests/ --profile black
pytest

# CLI
leaps-preprocess    # build HDF5 dataset from raw Camargo .mat files
leaps-train-strides # stride-level model training (primary)
leaps-simulate      # run simulation
leaps-render        # render comparison (generate on cluster, render locally)
leaps-visualize     # visualize EMG activations
```

## Code Style
- black line-length 100, isort black profile
- snake_case / PascalCase / UPPER_SNAKE_CASE
- Absolute imports within package; `__all__` in every `__init__.py`
- Python 3.9+ type hints (`list[int]` not `List[int]`)
- Tests in `tests/`, pytest with `--cov=src/leaps`

## Data
- Raw: Camargo dataset at `/fast/lsivakumar/datasets/camargo/` (cluster only)
- Processed: `/fast/lsivakumar/data/processed/emg_activations_v2.h5`
- Local Windows clone: `C:\Users\lsivakumar\Documents\LEAPS`
- MATLAB `.m` reference scripts on cluster — Python pipeline replicates their logic
