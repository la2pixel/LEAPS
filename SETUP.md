# New Server Setup

## What's where

```
src/leaps/
├── data/           EMG preprocessing, HDF5 loading, PyTorch datasets — DO NOT TOUCH
├── models/         AE models (HausdorferAE, StrideFlatAE, NMF, PCA) — DO NOT TOUCH
├── training/       Stride-level training loop — DO NOT TOUCH
├── envs/           TO BE REWRITTEN — only emg_mapping.py is still valid
│   └── emg_mapping.py   EMG→muscle bilateral mapping, simulator-agnostic — KEEP
└── scripts/        CLI entry points (see Commands below)

experiments/old/    All old MuJoCo/PPO experiments — reference only
eda_plots/          All diagnostic plots from old runs
renders/old/        Two PPO videos from old runs
.claude/            Project docs: Architecture.md, Status.md, Papers.md
```

**Processed data** lives outside the repo. Set env vars before anything else (see below).

---

## Step 1 — Environment variables

Add to `~/.bashrc` (or equivalent) on the new server:

```bash
export CAMARGO_DATA=/path/to/camargo/dataset     # raw .mat files
export LEAPS_EMG_H5=/path/to/emg_activations_v2.h5
```

Run `source ~/.bashrc` after.

---

## Step 2 — Python environment

```bash
cd /path/to/LEAPS
python3 -m venv .venv
source .venv/bin/activate

# Install LEAPS in editable mode (core deps: numpy, scipy, torch, h5py etc.)
pip install -e ".[dev,wandb]"

# Install sconegym from your local clone
pip install -e /path/to/sconegym

# Install dep-rl from your local clone (if needed at this stage)
pip install -e /path/to/dep-rl
```

---

## Step 3 — Verify data pipeline works

```bash
# Should print stride counts per mode — no errors means data + env vars are correct
python3 -c "
from leaps.data.metadata import LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides
import h5py, os
print('H5 path:', LEAPS_H5_PATH)
print('Exists:', os.path.exists(LEAPS_H5_PATH))
with h5py.File(LEAPS_H5_PATH) as f:
    for mode in f:
        n = sum(len(f[mode][s]) for s in f[mode])
        print(f'  {mode}: {n} strides')
"
```

---

## Step 4 — Verify sconegym + H0918

```bash
python3 -c "
import sconegym
env = sconegym.make('H0918-whatever-the-id-is')   # check sconegym docs for exact env id
obs, _ = env.reset()
print('obs shape:', obs.shape)
print('action space:', env.action_space)
env.close()
"
```

Confirm:
- Action space matches 18 muscles (or however many H0918 exposes)
- Observation space includes joint positions and velocities
- Control frequency (check sconegym docs — should be ~100Hz to match EMG 101-point stride)

---

## Step 5 — Train the AE priors (no simulation needed)

```bash
# Snapshot AE (spatial prior — Hausdörfer-exact, trains in minutes)
leaps train-snapshot --epochs 300 --wandb

# Stride AE (temporal prior — LEAPS contribution)
leaps train-strides --models StrideFlatAE --latent-dims 8 --epochs 200 --wandb
```

Checkpoints saved to `experiments/ae_priors/` by default.
Target: HausdorferAE latent pct_inside_±0.8 ≥ 90%, StrideFlatAE R² ≥ 0.63.

---

## Step 6 — Build the sconegym env wrapper (Phase 0 → Phase 2 in Status.md)

Files to create (don't exist yet — start from scratch):

```
src/leaps/envs/scone_env.py      LeapsSconeEnv(gym.Env) — wraps sconegym H0918
src/leaps/envs/latent_env.py     LatentSconeEnv — policy outputs z, frozen decoder maps to muscles
```

Key things to figure out first:
1. Muscle order in sconegym H0918 — must match `emg_mapping.py` output order
2. Whether sconegym exposes GRF and joint-limit torques (needed for Schumacher reward)
3. Exact gym env ID string for H0918

---

## Commands

```bash
leaps preprocess      # rebuild HDF5 from raw Camargo .mat files (if needed)
leaps train-snapshot  # train HausdorferAE (spatial prior)
leaps train-strides   # train StrideFlatAE / sweep (temporal prior + baselines)
leaps visualize       # EMG pipeline diagnostic plots (needs CAMARGO_DATA)
leaps visualize-snapshot  # AE reconstruction quality plots (needs trained checkpoint)
```

---

## Key docs

| File | What's in it |
|---|---|
| `.claude/Status.md` | Full micro-step roadmap (Phases 0–9) |
| `.claude/Architecture.md` | Pipeline, AE architectures, EMG→muscle mapping details |
| `.claude/Papers.md` | Hausdörfer, Schumacher, Lattice — exact hyperparameters |
| `src/leaps/envs/emg_mapping.py` | EMG→18-muscle bilateral mapping (still valid) |
| `src/leaps/models/stride_models.py` | All AE models |
| `experiments/old/` | Old MuJoCo experiments — reference only |
