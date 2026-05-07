# LEAPS

EMG-based latent action priors for muscle-actuated humanoid locomotion.

Plug in any gait EMG dataset → physiologically grounded, sample-efficient RL agent via the LEAPS framework.

**Simulator**: SCONE + Hyfydy via sconegym | **Models**: H0918 (2D, 18 muscles), MyoLeg (3D)

---

## Setup

See [SETUP.md](SETUP.md) for full instructions. Short version:

```bash
# Set data paths
export LEAPS_RAW_DATA=/path/to/camargo
export LEAPS_PROCESSED_DATA=/path/to/emg_activations_v2.h5

# Install
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,wandb]"
pip install -e /path/to/sconegym
```

## Commands

```bash
leaps-preprocess        # build HDF5 from raw Camargo .mat files
leaps-train-snapshot    # train HausdorferAE (spatial prior)
leaps-train-strides     # train StrideFlatAE / sweep (temporal prior + baselines)
leaps-visualize         # EMG pipeline diagnostic plots
leaps-visualize-snapshot  # AE reconstruction quality
```

## License

MIT
