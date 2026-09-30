# LEAPS

Code for *Learning Humanoid Locomotion from EMG-Based Latent Action Priors and Muscle Synergies* (MSc thesis, 2026).

A small autoencoder is trained on surface EMG from one human subject (Camargo et al. 2021, AB06, 11 lower-limb channels). Its frozen decoder becomes part of the environment: the RL policy outputs a latent `z` plus a residual, the decoder turns `z` into EMG-like muscle excitations, and the residual corrects them with weight `w`. It is evaluated with MPO and DEP-MPO on three musculoskeletal models in SCONE/Hyfydy: H0918 (2D, 18 muscles), H1622 (3D, 22) and H2190 (3D, 90).

## Setup

Requirements:
- Linux
- [SCONE](https://scone.software) with a [Hyfydy](https://hyfydy.com) license (sconepy needs Python 3.9)
- conda

Then run:

```bash
source setup.sh
```

`setup.sh` does three things:
- creates the `leaps` conda env
- clones the two forks this code depends on into `deps/`, pinned to the commits used for the thesis:
  - [sconegym](https://github.com/la2pixel/sconegym) for the H0918/H1622/H2190 velocity-reward environments
  - [depRL](https://github.com/la2pixel/depRL) for MPO/DEP-MPO; it imports `leaps.envs`
- installs everything in editable mode

Environment variables:

| variable | what | needed for |
|---|---|---|
| `CAMARGO_DATA` | raw [Camargo et al. 2021](https://doi.org/10.1016/j.jbiomech.2021.110320) dataset | rebuilding the EMG h5 |
| `LEAPS_EMG_H5` | processed EMG strides (`leaps preprocess` output) | training decoders, comparisons against human EMG |
| `SCONE_LIVE_DIR` | where Hyfydy writes training runs (default `~/Documents/SCONE/results/live`) | evaluation scripts |

RL training only needs the released decoders in `priors/`. It does not need the dataset.

## Repository layout

```
priors/                      frozen EMG decoders, AB06, k = 2, 4, 6, 8, 11 (k=6 is the one used throughout)
  decoder_k*_AB06_corrected/ decoder.pt, norm.npz (latent/EMG ranges), results.yaml (fit + held-out error)
src/leaps/
  data/                      Camargo parsing, EMG filtering/normalization, stride segmentation -> h5
  envs/                      latent_env.py (the prior wrapper), EMG->muscle mapping, control priors
  configs/                   RunSpec -> deprl config.yaml generator
  models/, training/         autoencoders and synergy extraction
  scripts/                   pipeline entry points (also exposed through the `leaps` CLI)
  analysis/                  scripts behind the thesis figures and tables
tests/
run.sh                       every experiment in the thesis, in order
```

## Reproducing the experiments

`run.sh` lists each step:
1. preprocessing
2. decoder training
3. config generation for every RL condition
4. training
5. evaluation

Config generation looks like this:

```bash
leaps launch --body h0918 h1622 h2190 --seeds 0 1 2    # the main setting: k=6, w=0.5
leaps launch-explore --category emg_lap --body h0918 --w 0.0 0.1 0.5 --seeds 0 1 2
```

Each command writes `baselines_DEPRL/final_experiments/<category>/<body>/<run>/config.yaml` and prints the matching `python -m deprl.main <config.yaml>`. Each run is 20M environment steps.

The conditions are:

| category | action space | prior |
|---|---|---|
| `mpo`, `dep-mpo` | muscle excitations | none |
| `emg_lap` | latent `z` + residual | EMG decoder |
| `no_lap` | same | `a_hat = 0` (null prior) |
| `untrained_lap` | same | randomly initialized decoder |

Results are reported per `w` ∈ {0, 0.1, 0.5}. The thesis used 3 seeds per condition, with fewer for some H2190 conditions. A few runs were extended past 20M steps. The evaluation always uses the checkpoint picked by `leaps select-checkpoint`, not the last one.

## Tests

```bash
pytest
```

## License

MIT
