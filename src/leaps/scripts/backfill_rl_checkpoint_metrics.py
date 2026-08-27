"""Roll out a saved deprl checkpoint and extract per-channel/per-muscle
metrics that were never logged live during training (2026-08-11: the fix in
latent_env.py::_diagnostics_post_clip only covers runs launched after that
date -- everything before it needs reconstructing from checkpoints, this
script is that reconstruction).

Reuses the exact agent/environment-from-config pattern deprl/play.py::play()
uses (eval(config["tonic"]["agent"]/["environment"])), not a hand-rebuilt
wrapper -- so it works for both LatentActionPriorWrapper runs and plain
mpo/dep-mpo baselines without branching.

Output is long-format CSV, one row per (run, seed, checkpoint_step, metric,
name, value) -- built for filtering/groupby later, same philosophy as
RUNS_INDEX.csv, not a fixed wide table that has to be redesigned per question.

Usage:
    python -m leaps.scripts.backfill_rl_checkpoint_metrics \\
        --config baselines_DEPRL/emg_lap/h0918/h0918_k11_w01_mirror_clip_net256_full_seed0/config.yaml \\
        --checkpoint-step 20000000 --episodes 5 --out results/backfill/h0918_hardconstraint11.csv

    # sweep every saved checkpoint for one run (temporal-drift analysis):
    python -m leaps.scripts.backfill_rl_checkpoint_metrics \\
        --config ... --sweep-checkpoints --episodes 3 --out ...
"""
import argparse
import contextlib
import glob
import io
import os
import re

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import pandas as pd
import torch
import yaml

torch.set_default_device("cpu")

LIVE = "/media/calc_2/scone_results/live"


def find_result_dir(tonic_name):
    d = os.path.join(LIVE, tonic_name)
    subs = [s for s in glob.glob(d + "/*") if os.path.isdir(s)]
    if not subs:
        raise FileNotFoundError(f"no result dir under {d}")
    # newest by mtime of its own checkpoints/ dir, matches convention used
    # everywhere else this project resolves a tonic.name to its real data
    return max(subs, key=lambda s: os.path.getmtime(s))


def list_checkpoint_steps(result_dir):
    ckpts = glob.glob(os.path.join(result_dir, "checkpoints", "step_*.pt"))
    steps = sorted(int(re.search(r"step_(\d+)\.pt", c).group(1)) for c in ckpts)
    return steps


def build_env(config):
    """Exact pattern from deprl/play.py::play() -- eval() the config's own
    environment string, not a hand-rebuilt version, so this works identically
    for wrapper and plain-baseline configs. env_wrappers.apply_wrapper() is
    what actually adds merge_args/apply_args to the env (SconeWrapper) --
    skipping it is why a first version of this script crashed on that call.

    Returns (outer_environment, inner_environment). gym.Wrapper.__getattr__
    deliberately raises on any name starting with "_" (core.py:213 -- "attempted
    to get missing private attribute"), so LatentActionPriorWrapper's _last_emg/
    _mapped_mask/etc. are NOT reachable through the outer SconeWrapper apply_wrapper()
    adds. inner_environment is the pre-wrap reference, kept because .step() on the
    outer wrapper still calls through to it internally (same object, same instance
    attributes), so reading state off inner_environment after stepping outer_environment
    is correct, not stale. No agent/checkpoint involved -- use this alone for anything
    that only needs env structure (e.g. actuator/mapping counts).
    """
    from deprl import env_wrappers
    tonic = config["tonic"]
    if tonic.get("header"):
        exec(tonic["header"], globals())
    inner_environment = eval(tonic["environment"])
    inner_environment.seed(tonic["seed"])
    outer_environment = env_wrappers.apply_wrapper(inner_environment)
    if config.get("env_args"):
        outer_environment.merge_args(config["env_args"])
        outer_environment.apply_args()
    return outer_environment, inner_environment


def build_agent_and_env(config):
    """build_env() plus the agent, initialized against that env's spaces.
    Returns (agent, outer_environment, inner_environment)."""
    outer_environment, inner_environment = build_env(config)
    tonic = config["tonic"]
    agent = eval(tonic["agent"])
    agent.initialize(
        observation_space=outer_environment.observation_space,
        action_space=outer_environment.action_space,
        seed=tonic["seed"],
    )
    # deprl/main.py applies config['mpo_args'] (hidden_size, lr_*, etc.) via
    # this exact call after construction -- skipping it silently/loudly
    # fails to load any checkpoint from a non-default net width (e.g.
    # net512): shapes mismatch, load raises, and rollout proceeds on an
    # untrained network with no further warning. Same gotcha documented in
    # leaps/models/extract_rollout_z_distribution.py -- fixed here too.
    if "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    return agent, outer_environment, inner_environment


def load_checkpoint_or_raise(agent, ckpt):
    """agent.load() swallows its own exceptions (tonic's Agent.load(),
    vendored, logs "Error, not loading model" and silently continues on a
    random-init network -- bit us once already on a net512 checkpoint
    with no mpo_args applied, see build_agent_and_env). Raise instead."""
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        agent.load(ckpt, only_checkpoint=True)
    if "Error, not loading model" in log.getvalue():
        print(log.getvalue())
        raise RuntimeError(
            f"agent.load({ckpt!r}) silently failed (see traceback above) -- "
            "check for a shape mismatch (e.g. missing/wrong mpo_args.hidden_size in the config)."
        )


def is_wrapper_env(inner_environment):
    return hasattr(inner_environment, "decoder") and hasattr(inner_environment, "_mapped_mask")


def rollout(agent, outer_environment, inner_environment, n_episodes, checkpoint_step, run_label, seed):
    """One rollout. Step/reset go through outer_environment (SconeWrapper --
    applies clip_actions etc.); per-step diagnostic state is read off
    inner_environment (LatentActionPriorWrapper), since gym.Wrapper refuses to
    delegate underscore-prefixed attributes -- see build_agent_and_env's docstring."""
    rows = []
    is_wrapped = is_wrapper_env(inner_environment)

    if is_wrapped:
        from leaps.envs.emg_mapping import EMG_CHANNELS
        all_names = [a.name() for a in outer_environment.unwrapped.model.actuators()]
        unmapped_mask = ~inner_environment._mapped_mask
        unmapped_names = [n for n, m in zip(all_names, unmapped_mask) if m]
        emg_acc = []
        prior_abs_acc = {n: [] for n in inner_environment._mapped_names}
        residual_unmapped_acc = {n: [] for n in unmapped_names}
    else:
        all_names = [a.name() for a in outer_environment.unwrapped.model.actuators()]
        activation_acc = {n: [] for n in all_names}

    scores = []
    obs = outer_environment.reset()
    ep = 0
    steps = 0
    ep_score = 0.0
    while ep < n_episodes:
        actions = agent.test_step(obs, muscle_states=None, steps=1e6)
        if len(actions.shape) > 1:
            actions = actions[0, :]
        obs, rew, done, info = outer_environment.step(actions)
        steps += 1
        ep_score += rew

        if is_wrapped:
            emg_acc.append(inner_environment._last_emg.copy())
            a_hat_clipped = np.clip(inner_environment._last_a_hat, 0.0, 0.5)
            final_clipped = np.clip(inner_environment._last_final_action, 0.0, 0.5)
            for name, val in zip(inner_environment._mapped_names, a_hat_clipped[inner_environment._mapped_mask]):
                prior_abs_acc[name].append(abs(val))
            for name, val in zip(unmapped_names, final_clipped[unmapped_mask]):
                residual_unmapped_acc[name].append(val)
        else:
            for name, val in zip(all_names, actions):
                activation_acc[name].append(val)

        if done or steps >= outer_environment.max_episode_steps:
            scores.append(ep_score)
            ep += 1
            obs = outer_environment.reset()
            steps = 0
            ep_score = 0.0

    def add_rows(kind, d):
        for name, vals in d.items():
            rows.append(dict(
                run=run_label, seed=seed, checkpoint_step=checkpoint_step,
                kind=kind, name=name, mean=float(np.mean(vals)), std=float(np.std(vals)),
                n_steps=len(vals),
            ))

    if is_wrapped:
        emg_arr = np.array(emg_acc)
        add_rows("decoded_emg", {c: emg_arr[:, i] for i, c in enumerate(EMG_CHANNELS)})
        add_rows("prior_abs_mapped", prior_abs_acc)
        add_rows("residual_unmapped", residual_unmapped_acc)
    else:
        add_rows("raw_activation", activation_acc)

    for i, s in enumerate(scores):
        rows.append(dict(run=run_label, seed=seed, checkpoint_step=checkpoint_step,
                          kind="episode_score", name=f"ep{i}", mean=float(s), std=0.0, n_steps=1))

    return rows


def run_one(config_path, checkpoint_step=None, episodes=5, sweep_checkpoints=False, run_label=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    tonic_name = config["tonic"]["name"]
    seed = config["tonic"]["seed"]
    run_label = run_label or os.path.basename(os.path.dirname(config_path))
    result_dir = find_result_dir(tonic_name)

    steps_to_run = list_checkpoint_steps(result_dir) if sweep_checkpoints else [
        checkpoint_step or list_checkpoint_steps(result_dir)[-1]
    ]

    all_rows = []
    for step in steps_to_run:
        ckpt = os.path.join(result_dir, "checkpoints", f"step_{step}")
        agent, outer_environment, inner_environment = build_agent_and_env(config)
        load_checkpoint_or_raise(agent, ckpt)
        rows = rollout(agent, outer_environment, inner_environment, episodes, step, run_label, seed)
        all_rows.extend(rows)
        print(f"  {run_label} seed{seed} step{step}: {len(rows)} rows "
              f"(scores={[round(r['mean']) for r in rows if r['kind']=='episode_score']})")

    return all_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a run's config.yaml")
    p.add_argument("--checkpoint-step", type=int, default=None, help="default: latest saved")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--sweep-checkpoints", action="store_true", help="run every saved checkpoint, for temporal-drift analysis")
    p.add_argument("--run-label", default=None, help="default: config's parent dir name")
    p.add_argument("--out", required=True, help="output CSV path -- appended to if it already exists")
    args = p.parse_args()

    rows = run_one(args.config, args.checkpoint_step, args.episodes, args.sweep_checkpoints, args.run_label)
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        df = pd.concat([pd.read_csv(args.out), df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(rows)} new rows, {len(df)} total, to {args.out}")


if __name__ == "__main__":
    main()
