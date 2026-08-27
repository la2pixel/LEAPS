# LEAPS

EMG-based latent action priors for muscle-actuated humanoid locomotion.

**Simulator**: SCONE + Hyfydy via sconegym | **Models**: H0918 (2D, 18 muscles)

---

## Setup

See [SETUP.md](SETUP.md) for full instructions. Short version:

```bash
conda activate lalitha

# Set data paths (add to env.sh)
export CAMARGO_DATA=/path/to/camargo
export LEAPS_EMG_H5=/path/to/emg_activations_v2.h5

pip install -e ".[dev,wandb]"
pip install -e /path/to/sconegym
pip install -e /path/to/depRL
```

## Commands

```bash
leaps preprocess        # build HDF5 from raw Camargo .mat files
leaps train-snapshot    # train HausdorferAE (spatial prior)
leaps train-strides     # train StrideFlatAE / sweep (temporal prior + baselines)
leaps visualize         # EMG pipeline diagnostic plots
leaps visualize-snapshot  # AE reconstruction quality
leaps --help            # full command list
```

## License

MIT
