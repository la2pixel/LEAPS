"""Repo-relative directories. Data paths (Camargo, EMG h5) are env vars, see data/metadata.py."""

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("LEAPS_ROOT", Path(__file__).resolve().parents[2]))
PRIORS_DIR = REPO_ROOT / "priors"  # released decoders, one dir per k
RESULTS_DIR = REPO_ROOT / "results"  # analysis outputs (gitignored)
RUNS_DIR = REPO_ROOT / "baselines_DEPRL"  # generated RL configs + checkpoints (gitignored)
FIGURES_DIR = RESULTS_DIR / "figures"
# where Hyfydy/deprl write training runs (tonic name "lalitha/<group>/..." is appended)
SCONE_LIVE_DIR = Path(os.environ.get("SCONE_LIVE_DIR", Path.home() / "Documents" / "SCONE" / "results" / "live"))
