### Note: This repo is still being updated!

[![tests](https://github.com/la2pixel/LEAPS/actions/workflows/tests.yml/badge.svg)](https://github.com/la2pixel/LEAPS/actions/workflows/tests.yml)

## LEAPS- Learning Humanoid Locomotion from EMG-Based Latent Action Priors and Muscle Synergies

📄 [Thesis](assets/thesis.pdf) | 🎞️ [Thesis defense slides](assets/thesis_defense.pptx)

![Method overview](assets/method_overview.png)

Results of our approach vs. standard frameworks on best checkpoints:

| H0918 | H1622 | H2190 |
|:---:|:---:|:---:|
| ![LEAPS on H0918](assets/videos/leaps_h0918.gif) | ![LEAPS on H1622](assets/videos/leaps_h1622.gif) | ![LEAPS on H2190](assets/videos/leaps_h2190.gif) |

On H2190, compared with the baselines at the best checkpoint:

| DEP-MPO | MPO | LEAPS |
|:---:|:---:|:---:|
| ![DEP-MPO on H2190](assets/videos/depmpo_h2190.gif) | ![MPO on H2190](assets/videos/mpo_h2190.gif) | ![LEAPS on H2190](assets/videos/leaps_h2190.gif) |

**Priors**: 
- Access the priors trained offline in `src/leaps/envs/latent_env.py`. Please note that MPO and DEP-MPO do not use them.
- Muscles without a corresponding EMG channel are driven by the residual alone. 

**Access**: This repository uses `conda` to manage the python environment. Run `source setup.sh` to set up the repository; it also clones the [sconegym](https://github.com/la2pixel/sconegym) and [depRL](https://github.com/la2pixel/depRL) forks used here. `priors` contains the released EMG decoders and how they were trained. `src/leaps` contains the EMG preprocessing, the latent action prior wrapper, the config generator for the RL runs and the evaluation scripts; most steps run through the `leaps` command. For examples on how to train the decoders and run the experiments see `run.sh`. If you are getting started with this repository, you probably want to look at `run.sh`.

**Important**: You require [SCONE](https://scone.software) with a [Hyfydy](https://hyfydy.com) license to run the simulations. RL training only needs the decoders in `priors`; retraining them requires the [Camargo et al. 2021](https://doi.org/10.1016/j.jbiomech.2021.110320) dataset (`CAMARGO_DATA`). Experiments are logged to `wandb`.
