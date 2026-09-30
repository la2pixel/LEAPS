#!/usr/bin/env bash
# Reproduces the thesis experiments, in order. Each step is independent; run
# the ones you need. RL runs are long (20M steps each), so launch
# the printed `python -m deprl.main <config>` commands with your own scheduler.
set -euo pipefail

# Set these first (see README):
#   CAMARGO_DATA   raw Camargo et al. 2021 dataset (only needed for step 1)
#   LEAPS_EMG_H5   processed EMG strides, output of step 1
#   SCONE_LIVE_DIR where Hyfydy writes training runs (default ~/Documents/SCONE/results/live)
# and log in to wandb if you want curves there.

# 1. EMG preprocessing: Camargo .mat -> filtered, normalized, 101-point strides in one h5.
#    Needed for step 2 and the step-5 comparisons against human EMG, not for RL training.
# leaps preprocess --output "$LEAPS_EMG_H5"

# 2. EMG decoders. The released ones in priors/ were made with exactly this;
#    rerunning overwrites them.
# for k in 2 4 6 8 11; do
#   python -m leaps.scripts.train_single_subject_decoder --subject AB06 --k $k \
#     --threshold 0.6 --scale 0.9 --depth 1 --seed 0 --suffix _corrected
# done

# 3. RL configs. Everything below only writes baselines_DEPRL/final_experiments/**/config.yaml
#    and prints the training commands.
BODIES="h0918 h1622 h2190"

# 3a. backbones
leaps launch-explore --category mpo dep-mpo --body $BODIES --seeds 0 1 2

# 3b. LEAPS, residual weight w in {0, 0.1, 0.5}
leaps launch-explore --category emg_lap --body $BODIES --w 0.0 0.1 0.5 --seeds 0 1 2

# 3c. controls: null prior (a_hat = 0) and untrained (random) decoder
leaps launch-explore --category no_lap --body $BODIES --w 0.5 --seeds 0 1 2
leaps launch-explore --category untrained_lap --body $BODIES --w 0.0 0.1 0.5 --seeds 0 1

# 3d. latent dimension sweep (h0918, w=0.1)
leaps launch-explore --category emg_lap --body h0918 --k 2 4 8 11 --w 0.1 --seeds 0 1 2

# 3e. full reward (effort/GRF/joint-limit/smoothness terms) instead of velocity only
leaps launch-explore --category mpo --body h0918 h1622 --reward-variant full --seeds 0 1 2
leaps launch-explore --category emg_lap --body h0918 h1622 --w 0.1 --reward-variant full --seeds 0 1 2

# 4. Train: python -m deprl.main <config.yaml>, for every config printed above.

# 5. Evaluation, after training
#    t* checkpoint per run, then fresh N=100 rollouts at t*
leaps select-checkpoint --help
leaps reeval-checkpoint --help
#    gait mechanics, muscle-activation fidelity vs human EMG, joint kinematics
leaps gait-metrics
leaps biomech-fidelity --figure f2 --tstar
leaps kinematic-match

# 6. Thesis figures/tables: python -m leaps.analysis.<script>, outputs in results/figures/
