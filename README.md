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
| `CAMARGO_DATA` | raw [Camargo et al. 2021](https://doi.org/10.1016/j.jbiomech.2021.110320) dataset | rebuilding the EMG h5, `kinematics_correlation` |
| `LEAPS_EMG_H5` | processed EMG strides (`leaps preprocess` output) | training decoders, comparisons against human EMG |
| `SCONE_LIVE_DIR` | where Hyfydy writes training runs (default `~/Documents/SCONE/results/live`) | evaluation scripts |

RL training only needs the released decoders in `priors/`. It does not need the dataset.

## Repository layout

```
priors/                      frozen EMG decoders, AB06, k = 2, 4, 6, 8, 11 (k=6 is the one used throughout)
  decoder_k*_AB06_corrected/ decoder.pt, encoder.pt, norm.npz, results.yaml, training + PCA plots (see priors/README.md)
src/leaps/
  data/                      Camargo parsing, EMG filtering/normalization, stride segmentation -> h5
  envs/                      latent_env.py (the prior wrapper), EMG->muscle mapping, control priors
  configs/                   RunSpec -> deprl config.yaml generator
  models/                    synergy extraction, rollout extractors
  scripts/                   pipeline entry points (also exposed through the `leaps` CLI)
  analysis/                  scripts behind the thesis figures and tables
docs/eda/                    EMG preprocessing diagnostics (leaps eda): seven figures + summary.yaml with stride counts
results/                     result tables behind the thesis numbers (see results/README.md); other outputs land here too, untracked
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

Each command writes `baselines_DEPRL/final_experiments/<category>/<body>/<run>/config.yaml` and prints the matching `python -m deprl.main <config.yaml>`. Each run is 20M environment steps (10M for the full-reward runs on H0918; `--steps` sets it).

The conditions are:

| category | action space | prior |
|---|---|---|
| `mpo`, `dep-mpo` | muscle excitations | none |
| `emg_lap` | latent `z` + residual | EMG decoder |
| `no_lap` | same | `a_hat = 0` (null prior) |
| `untrained_lap` | same | randomly initialized decoder |

Results are reported per `w` ∈ {0, 0.1, 0.5}. `run.sh` lists the seeds trained per condition (2 to 4). The per-seed files in `results/checkpoint_selection/` show which seeds entered the return table, and `RUNS` in `src/leaps/scripts/biomech_fidelity.py` which runs the gait and EMG analyses read. A few runs were trained past 20M steps; checkpoint selection only considers the first 20M. The evaluation always uses the checkpoint picked by `leaps select-checkpoint`, not the last one. `run.sh` step 5 has the commands for checkpoint selection, re-evaluation, rollouts at the selected checkpoint (`deprl.play`) and the gait and EMG comparisons.

## Results

All numbers come from the tables in `results/` (see `results/README.md`); the thesis has the full analysis.

Return at the selected checkpoint, 100 rollouts per seed: mean ± SD across seeds, with the percentage of rollouts ending in a fall. Velocity-only reward, maximum return 1000:

| condition | H0918 | H1622 | H2190 |
|---|---|---|---|
| MPO | 999 ± 0 (0%) | 995 ± 5 (0%) | 505 ± 28 (74%) |
| DEP-MPO | 999 ± 0 (0%) | 995 ± 4 (0%) | 519 ± 83 (73%) |
| LEAPS, w = 0 | 1000 ± 0 (0%) | 957 ± 3 (4%) | 402 ± 31 (88%) |
| LEAPS, w = 0.1 | 1000 ± 0 (0%) | 902 ± 81 (12%) | 342 ± 40 (91%) |
| LEAPS, w = 0.5 | 1000 ± 0 (0%) | 986 ± 2 (2%) | 440 ± 69 (70%) |
| null prior | 1000 ± 0 (0%) | 967 ± 17 (4%) | 351 ± 19 (90%) |
| untrained decoder, w = 0.5 | 999 ± 0 (0%) | 998 ± 1 (0%) | 606 ± 90 (63%) |

Seeds per cell: 3 on H0918 and 2 to 4 on H1622 and H2190 (2 for the untrained decoder). The untrained decoder was redrawn at every environment build, so it is not a fixed random decoder.

Full reward:

| condition | H0918 | H1622 |
|---|---|---|
| MPO | 9600 ± 10 (0%) | 8573 ± 213 (1%) |
| LEAPS, w = 0.1 | 9218 ± 82 (0%) | 8109 ± 324 (6%) |

Muscle-activation timing: Pearson r over the gait cycle between the simulated activation and the EMG of the 22 Camargo subjects (mean over subjects). The `human` column is the mean r between pairs of subjects. Policies are evaluated at the selected checkpoint, except H0918 with w = 0.5, which is at 20M steps (see `results/README.md`).

H0918:

| muscle | human | MPO | DEP-MPO | w = 0 | w = 0.1 | w = 0.5 | null |
|---|---|---|---|---|---|---|---|
| SOL | 0.90 | 0.88 | 0.80 | -0.02 | 0.00 | 0.79 | 0.69 |
| GAS | 0.89 | 0.64 | 0.65 | -0.01 | -0.06 | 0.77 | 0.58 |
| TA | 0.80 | 0.64 | 0.52 | -0.33 | -0.38 | 0.49 | 0.61 |
| ST | 0.69 | 0.39 | 0.37 | 0.31 | 0.09 | 0.25 | 0.34 |
| BF | 0.64 | 0.42 | 0.42 | 0.38 | 0.12 | 0.28 | 0.34 |
| RF | 0.74 | -0.47 | 0.04 | -0.49 | -0.32 | 0.09 | -0.38 |
| VM | 0.90 | 0.61 | 0.61 | 0.32 | 0.41 | 0.14 | 0.60 |
| VL | 0.90 | 0.62 | 0.64 | 0.32 | 0.45 | 0.16 | 0.63 |

H1622:

| muscle | human | MPO | DEP-MPO | w = 0 | w = 0.1 | w = 0.5 | null |
|---|---|---|---|---|---|---|---|
| SOL | 0.90 | 0.72 | 0.38 | 0.28 | 0.39 | 0.60 | 0.91 |
| GAS | 0.89 | 0.68 | 0.65 | 0.17 | 0.33 | 0.66 | 0.75 |
| TA | 0.80 | 0.60 | 0.47 | 0.26 | 0.32 | 0.26 | 0.57 |
| ST | 0.69 | -0.24 | 0.23 | 0.79 | 0.71 | 0.01 | -0.13 |
| BF | 0.64 | -0.17 | 0.29 | 0.76 | 0.72 | 0.09 | -0.07 |
| RF | 0.74 | -0.08 | 0.51 | -0.18 | -0.24 | 0.03 | 0.17 |
| VM | 0.90 | 0.75 | 0.62 | 0.47 | 0.72 | 0.51 | 0.85 |
| VL | 0.90 | 0.74 | 0.61 | 0.50 | 0.75 | 0.50 | 0.84 |

H2190:

| muscle | human | MPO | DEP-MPO | w = 0 | w = 0.1 | w = 0.5 | null |
|---|---|---|---|---|---|---|---|
| SOL | 0.90 | 0.60 | 0.60 | 0.72 | 0.35 | 0.24 | 0.74 |
| GAS | 0.89 | 0.86 | 0.51 | 0.61 | -0.05 | 0.22 | 0.68 |
| TA | 0.80 | 0.56 | 0.39 | 0.11 | -0.22 | 0.24 | 0.34 |
| ST | 0.69 | -0.36 | -0.16 | -0.02 | 0.45 | -0.31 | -0.60 |
| BF | 0.64 | -0.07 | -0.22 | 0.26 | 0.61 | -0.04 | 0.24 |
| RF | 0.74 | 0.25 | 0.23 | -0.74 | -0.66 | 0.02 | -0.14 |
| VM | 0.90 | 0.53 | 0.68 | 0.00 | 0.67 | 0.74 | 0.40 |
| VL | 0.90 | 0.54 | 0.59 | 0.08 | 0.72 | 0.75 | 0.52 |

On H1622, the policies with w = 0 and w = 0.1 mostly hop with stiff knees instead of walking, and on H2190 most rollouts end in a fall for every method; `results/gait_metrics/GAIT_METRICS.csv` has the gait mechanics per condition.

## Tests

```bash
pytest
```

## License

MIT
