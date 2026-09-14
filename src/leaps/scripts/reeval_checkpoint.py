"""Phase-2 of the checkpoint-selection rule (locked 2026-09-08): fresh
N-episode deterministic eval at each seed's t* checkpoint.

`score` here = sum of per-step env reward over the episode = exactly the
quantity tonic logs as test/episode_score, so the output is directly
comparable to the training curve / select_checkpoint.py's score@t*.

Reads a selection CSV written by select_checkpoint.py (cols: label, seed,
t_star, score_at_tstar, final10, run_dir) and re-evals every row.

    python -m leaps.scripts.reeval_checkpoint \
        --selection LEAPS/results/checkpoint_selection/emg_lap_h1622_w01_onlyvelrew.csv \
        --episodes 100
"""
import argparse
import csv
import glob
import os
import statistics
import sys
import time

import numpy as np
import yaml


def find_ts_dir(parent, step):
    """The <timestamp>/ dir under `parent` whose checkpoints/ has step_<N>.pt."""
    for cfg in glob.glob(os.path.join(parent, "*", "config.yaml")):
        d = os.path.dirname(cfg)
        if os.path.exists(os.path.join(d, "checkpoints", f"step_{int(step)}.pt")):
            return d
    return None


def build_env_and_agent(ts_dir, step):
    from deprl import env_wrappers
    from deprl.utils import load_checkpoint

    cfg = yaml.safe_load(open(os.path.join(ts_dir, "config.yaml")))
    tc = cfg["tonic"]
    exec(tc["header"])  # noqa: S102

    _, ckpt_path, _ = load_checkpoint(os.path.join(ts_dir, "checkpoints"), str(step))
    if ckpt_path is None:
        raise SystemExit(f"checkpoint {step} not found in {ts_dir}")

    env = eval(tc["environment"])  # noqa: S307
    env.seed(0)
    env = env_wrappers.apply_wrapper(env)
    if cfg.get("env_args"):
        env.merge_args(cfg["env_args"])
        env.apply_args()

    agent = eval(tc["agent"])  # noqa: S307
    agent.initialize(observation_space=env.observation_space,
                     action_space=env.action_space, seed=0)
    agent.load(ckpt_path, only_checkpoint=True)
    return env, agent


def rollout(env, agent, episodes, base_seed=1000):
    cap = getattr(env, "max_episode_steps", 1000)
    scores, lengths, falls = [], [], 0
    for i in range(episodes):
        try:
            env.seed(base_seed + i)
        except Exception:
            pass
        obs = env.reset()
        ms = env.muscle_states
        score = length = 0.0
        done = False
        while not done and length < cap:
            a = agent.test_step(obs, muscle_states=ms, steps=1e6)
            a = a[0] if getattr(a, "ndim", 1) > 1 else a
            obs, r, done, _ = env.step(a)
            ms = env.muscle_states
            score += float(r)
            length += 1
        scores.append(score)
        lengths.append(int(length))
        if length < 0.9 * cap:
            falls += 1
    return scores, lengths, falls, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.selection)))
    out = args.out or args.selection.replace(".csv", "_reeval.csv")
    results = []
    print(f"re-eval  {args.selection}   {args.episodes} episodes/seed\n")
    hdr = f"{'seed':>6} {'t*(M)':>6} {'score@t* (curve)':>16} {'reeval mean':>12} {'sd':>7} {'sem':>6} {'fall%':>6} {'min':>7}"
    print(hdr); print("-" * len(hdr))

    for r in rows:
        parent, step = r["run_dir"], int(r["t_star"])
        ts = find_ts_dir(parent, step)
        if ts is None:
            print(f"{r['seed']:>6}  !! no checkpoint step_{step} under {parent}", file=sys.stderr)
            continue
        t0 = time.time()
        env, agent = build_env_and_agent(ts, step)
        scores, lengths, falls, cap = rollout(env, agent, args.episodes)
        m = statistics.mean(scores)
        sd = statistics.pstdev(scores)
        sem = sd / len(scores) ** 0.5
        fr = 100.0 * falls / len(scores)
        results.append(dict(label=r["label"], seed=r["seed"], t_star=step,
                            score_at_tstar_curve=r["score_at_tstar"],
                            reeval_mean=round(m, 1), reeval_sd=round(sd, 1),
                            reeval_sem=round(sem, 2), fall_pct=round(fr, 1),
                            reeval_min=round(min(scores), 1), episodes=len(scores)))
        print(f"{r['seed']:>6} {step/1e6:>6.0f} {float(r['score_at_tstar']):>16.1f} "
              f"{m:>12.1f} {sd:>7.1f} {sem:>6.1f} {fr:>6.1f} {min(scores):>7.1f}   ({time.time()-t0:.0f}s)")

    if results:
        agg = [x["reeval_mean"] for x in results]
        ma, sda = statistics.mean(agg), (statistics.pstdev(agg) if len(agg) > 1 else 0.0)
        print("-" * len(hdr))
        print(f"\nREPORT (re-eval @ t*):  {ma:.1f} +/- {sda:.1f}   over {len(agg)} seeds")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
