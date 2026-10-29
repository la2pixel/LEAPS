## Note: This code is being updated!

[![tests](https://github.com/la2pixel/LEAPS/actions/workflows/tests.yml/badge.svg)](https://github.com/la2pixel/LEAPS/actions/workflows/tests.yml)

### Code accompanying *Learning Humanoid Locomotion from EMG-Based Latent Action Priors and Muscle Synergies*

A small autoencoder is trained on surface EMG from one human subject (Camargo et al. 2021). Its frozen decoder becomes part of the environment: the RL policy outputs a latent `z` plus a residual, and the decoder turns `z` into muscle excitations. It is evaluated with MPO and DEP-MPO on three musculoskeletal models in SCONE/Hyfydy: H0918 (2D, 18 muscles), H1622 (3D, 22) and H2190 (3D, 90).

<!-- videos: LEAPS vs MPO / DEP-MPO -->

**How to use this repository**: This repository uses `conda` to manage the python environment. Run `source setup.sh` to set up the repository; it also clones the [sconegym](https://github.com/la2pixel/sconegym) and [depRL](https://github.com/la2pixel/depRL) forks used here. `priors` contains the released EMG decoders and how they were trained. `src/leaps` contains the EMG preprocessing, the latent action prior wrapper, the config generator for the RL runs and the evaluation scripts; most steps run through the `leaps` command. For examples on how to train the decoders and run the experiments see `run.sh`. If you are getting started with this repository, you probably want to look at `run.sh`.

**Important**: You require [SCONE](https://scone.software) with a [Hyfydy](https://hyfydy.com) license to run the simulations. RL training only needs the decoders in `priors`; retraining them requires the [Camargo et al. 2021](https://doi.org/10.1016/j.jbiomech.2021.110320) dataset (`CAMARGO_DATA`). Experiments are logged to `wandb`.
