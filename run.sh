#!/usr/bin/env bash
# Reproduces the thesis experiments, in order. Each step is independent; run
# the ones you need. RL runs are long (up to 20M steps each), so launch
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
#    Diagnostic figures for each preprocessing stage (filter chain, stride segmentation,
#    time-normalized strides, normalization, per-mode profiles). The tracked copies are in docs/eda/:
# leaps eda --data-root "$CAMARGO_DATA" --h5 "$LEAPS_EMG_H5" --subject AB06 --output-dir docs/eda --dpi 60
#    (add --norm priors/decoder_k6_AB06/norm.npz after step 2 for the decoder-input figure)

# 2. EMG decoders. The released ones in priors/ were made with exactly this;
#    rerunning overwrites them.
# for k in 2 4 6 8 11; do
#   python -m leaps.scripts.train_single_subject_decoder --subject AB06 --k $k \
#     --threshold 0.6 --scale 0.9 --depth 1 --seed 0
# done

# 3. RL configs. Everything below only writes baselines_DEPRL/final_experiments/**/config.yaml
#    and prints the training commands.
BODIES="h0918 h1622 h2190"

# 3a. backbones
leaps launch --category mpo dep-mpo --body $BODIES --seeds 0 1 2

# 3b. LEAPS, residual weight w in {0, 0.1, 0.5}. Seeds as trained for the thesis.
leaps launch --category emg_lap --body h0918 --w 0.0 0.1 0.5 --seeds 0 1 2
leaps launch --category emg_lap --body h1622 h2190 --w 0.1 --seeds 0 1 2 3
leaps launch --category emg_lap --body h1622 h2190 --w 0.0 0.5 --seeds 0 1

# 3c. controls: null prior (a_hat = 0) and untrained (random) decoder
leaps launch --category no_lap --body h0918 h1622 --w 0.5 --seeds 0 1 2
leaps launch --category no_lap --body h2190 --w 0.5 --seeds 0 1
leaps launch --category untrained_lap --body $BODIES --w 0.0 0.1 0.5 --seeds 0 1

# 3d. latent dimension sweep (h0918, w=0.1)
leaps launch --category emg_lap --body h0918 --k 2 4 8 11 --w 0.1 --seeds 0 1 2

# 3e. full reward (effort/GRF/joint-limit/smoothness terms) instead of velocity only.
#     H0918 was trained for 10M steps, H1622 for 20M.
leaps launch --category mpo emg_lap --body h0918 --w 0.1 --reward-variant full --steps 10e6 --seeds 0 1 2
leaps launch --category mpo emg_lap --body h1622 --w 0.1 --reward-variant full --seeds 0 1 2

# 4. Train: python -m deprl.main <config.yaml>, for every config printed above.

# 5. Evaluation, after training. Run dirs are under F; labels are free-form
#    names for one condition (all seeds), reused by the later steps.
F="${SCONE_LIVE_DIR:-$HOME/Documents/SCONE/results/live}/lalitha/final_experiments"
SEL=results/checkpoint_selection

# 5a. checkpoint t* per seed (budget = common horizon; runs trained longer are cut there),
#     then 100 fresh rollouts at t*. Repeat for every condition, e.g.:
# leaps select-checkpoint $F/emg_lap/h1622/h1622_k6_w01_onlyvelrew_seed{0,1,2} \
#   --label emg_lap_h1622_w01_onlyvelrew --budget 20000000
# leaps reeval-checkpoint --selection $SEL/emg_lap_h1622_w01_onlyvelrew.csv --episodes 100
#     Labels used for the thesis: {mpo,depmpo,null}_<body>_<reward>, emg_lap_<body>_<w00|w01|w05>_<reward>,
#     untrainedlap_<body>_w05_<reward>; H0918 full reward uses --budget 10000000.

# 5b. rollouts at t* (.sto files in <run>/<timestamp>/run_checkpoint_<t*>/), one call per seed
#     with t* from the selection csv:
# python -m deprl.play --path $F/emg_lap/h1622/<run>/<timestamp> --checkpoint <t*> --num_episodes 10

# 5c. gait mechanics, muscle-activation fidelity vs human EMG, joint kinematics.
#     These read the run dirs listed in RUNS in src/leaps/scripts/biomech_fidelity.py.
# leaps gait-metrics
# for b in $BODIES; do leaps biomech-fidelity --figure f2 --body $b --tstar --caption-order \\
#   --conditions mpo,dep-mpo,null,untrained,emg-w00,emg-w01,emg-w05; done
# leaps biomech-fidelity --figure ksweep           # latent dimension sweep, H0918
# leaps kinematic-match --regime onlyvel
# leaps kinematic-match --regime variants
# leaps kinematic-match --regime full
