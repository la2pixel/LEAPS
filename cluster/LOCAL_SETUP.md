# Local Rendering Setup (Windows)

The cluster is headless — all rendering happens locally.

## 1. Install Python + dependencies

```powershell
# Install Python 3.10+ from python.org if not already installed
# Then in PowerShell:
cd C:\Users\lsivakumar\Documents\LEAPS
pip install mujoco numpy imageio imageio-ffmpeg torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Only CPU PyTorch needed locally — GPU training happens on cluster.

## 2. Sync results from cluster

```powershell
# From your local machine (campus network or VPN):
scp -r cluster:"/fast/lsivakumar/LEAPS/experiments/stride_sweep/checkpoints/*.pt" experiments/stride_sweep/checkpoints/
```

Or use the shared drive path: `\\login.cluster.is.localnet\is\cluster\fast\lsivakumar\LEAPS\`

## 3. Render comparison video

```powershell
# Replay real EMG data (no model needed):
python -m leaps.scripts.render_comparison --data data/processed/emg_activations.h5 --mode replay --output renders/emg_replay.mp4

# With a trained model:
python -m leaps.scripts.render_comparison --checkpoint experiments/stride_sweep/checkpoints/StrideAutoencoder_d4.pt --model StrideAE --latent-dim 4 --output renders/comparison.mp4
```

## 4. Workflow

```
Cluster (GPU, headless)          Local (CPU, display)
─────────────────────────        ─────────────────────
condor_submit_bid 10
  cluster/train.sub
       │
  trains models, saves
  checkpoints to /fast
       │
       └──── scp/sync ──────►  python -m leaps.scripts.render_comparison
                                     │
                                  renders/comparison.mp4
```
