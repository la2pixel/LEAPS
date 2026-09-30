"""Single entry point for all LEAPS commands.

Usage:
    leaps <command> [args...]

Data:
    preprocess          Build EMG HDF5 from raw Camargo .mat files
    eda                 EMG pipeline diagnostic plots (raw → rectified → strides)

Training:
    train-strides       Train stride-level AE (primary LEAPS model)
    train-snapshot      Train HausdorferAE on per-frame snapshots

Config generation:
    launch              Generate configs for the locked-in production set (k=6, w=0.5, unclipped)
    launch-explore      Generate configs for any RunSpec knob combination, sweeps

Evaluation (after RL training):
    select-checkpoint   Pick t* per run from the training curves
    reeval-checkpoint   Re-evaluate t* checkpoints on fresh rollouts
    gait-metrics        Gait-mechanics battery per policy
    biomech-fidelity    Muscle activations vs cross-subject human EMG
    kinematic-match     Joint kinematics vs the Camargo band

Analysis:
    plot                Plot reconstruction quality across models and latent dims
    inspect-latent      Inspect trained model's latent space
    inspect-reward      Visualize reward landscape before training

Simulation:
    visualize           Visualize EMG preprocessing pipeline
    visualize-snapshot  Visualize snapshot AE reconstruction quality
"""

import importlib
import sys

_COMMANDS: dict[str, tuple[str, str]] = {
    # Data
    "preprocess":          ("leaps.data.build_h5",                  "main"),
    "eda":                 ("leaps.scripts.eda_emg",                "main"),
    # Training
    "train-strides":       ("leaps.scripts.train_strides",          "main"),
    "train-snapshot":      ("leaps.scripts.train_snapshot",         "main"),
    # Config generation
    "launch":              ("leaps.scripts.launch_locked",          "main"),
    "launch-explore":      ("leaps.scripts.launch_explore",         "main"),
    # Evaluation
    "select-checkpoint":   ("leaps.scripts.select_checkpoint",      "main"),
    "reeval-checkpoint":   ("leaps.scripts.reeval_checkpoint",      "main"),
    "gait-metrics":        ("leaps.scripts.gait_metrics",           "main"),
    "biomech-fidelity":    ("leaps.scripts.biomech_fidelity",       "main"),
    "kinematic-match":     ("leaps.scripts.kinematic_match",        "main"),
    # Analysis
    "plot":                ("leaps.scripts.plot_reconstructions",   "main"),
    "inspect-latent":      ("leaps.scripts.inspect_latent",         "main"),
    "inspect-reward":      ("leaps.scripts.inspect_reward",         "main"),
    # Simulation
    "visualize":           ("leaps.scripts.visualize_emg",          "main"),
    "visualize-snapshot":  ("leaps.scripts.visualize_snapshot_ae",  "main"),
}


def _help() -> None:
    print(__doc__)
    print("Run `leaps <command> --help` for command-specific options.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _help()
        return

    command = sys.argv[1]
    if command not in _COMMANDS:
        print(f"leaps: unknown command '{command}'\n")
        _help()
        sys.exit(1)

    # Strip the dispatch token so each script's argparser sees a clean argv.
    sys.argv = [f"leaps-{command}"] + sys.argv[2:]

    mod_name, func_name = _COMMANDS[command]
    mod = importlib.import_module(mod_name)
    getattr(mod, func_name)()
