"""Zero-shot held-out-speed eval for the velocity-conditioned runs (step 7).

Loads a trained checkpoint, pins env.target_vel to each speed in a grid
(no per-episode resampling), runs N episodes, reports the gaussianVel score
and the achieved-vs-target velocity error. Interpolation speeds sit inside
the trained U[0.8, 1.5] range, extrapolation speeds outside it.

The obs speed feature stays normalized the way training saw it
((v - 0.8) / 0.7, clipped) even for extrapolation speeds -- so 1.6/1.8 both
present to the policy as u=1. The decoder's own speed conditioning
(speed_lo/hi from norm.npz, ~0.5-2.05) is unclipped and still separates them.

    python -m leaps.scripts.eval_vcond_heldout_speed \
        --run-dir /media/calc_2/scone_results/live/lalitha/final_experiments/vcond/h0918/vcond_real_seed0/260827.120220.H0918v2j \
        --checkpoint 6000000 --episodes 20
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

TRAIN_LO, TRAIN_HI = 0.8, 1.5
DEFAULT_SPEEDS = [0.6, 0.9, 1.1, 1.3, 1.6, 1.8]


def build_env_and_agent(run_dir, checkpoint):
    from deprl import env_wrappers
    from deprl.utils import load_checkpoint

    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
    tc = cfg["tonic"]
    exec(tc["header"])  # noqa: S102 -- imports deprl, gym, sconegym, leaps.envs

    _, ckpt_path, _ = load_checkpoint(
        os.path.join(run_dir, "checkpoints"), str(checkpoint)
    )
    if ckpt_path is None:
        raise SystemExit(f"checkpoint {checkpoint} not found in {run_dir}")

    env = eval(tc["environment"])  # noqa: S307
    env.seed(0)
    env = env_wrappers.apply_wrapper(env)
    if cfg.get("env_args"):
        env.merge_args(cfg["env_args"])
        env.apply_args()

    agent = eval(tc["agent"])  # noqa: S307
    agent.initialize(
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=0,
    )
    agent.load(ckpt_path, only_checkpoint=True)
    return env, agent, cfg


def pin_speed(env, v):
    """Stop reset() resampling; keep training-consistent obs normalization."""
    import types

    u = env.unwrapped
    u.target_vel_range = None
    u.target_vel = float(v)

    def _fixed_tvoa(self):
        uu = np.clip((self.target_vel - TRAIN_LO) / (TRAIN_HI - TRAIN_LO), 0.0, 1.0)
        return np.array([uu], dtype=np.float32)

    u._target_vel_obs_array = types.MethodType(_fixed_tvoa, u)


def run_speed(env, agent, v, episodes):
    pin_speed(env, v)
    rows = []
    for _ in range(episodes):
        obs = env.reset()
        ms = env.muscle_states
        score = length = 0.0
        vels = []
        done = False
        while not done and length < env.max_episode_steps:
            a = agent.test_step(obs, muscle_states=ms, steps=1e6)
            a = a[0] if getattr(a, "ndim", 1) > 1 else a
            obs, r, done, _ = env.step(a)
            ms = env.muscle_states
            vels.append(float(env.unwrapped.model_velocity()))
            score += float(r)
            length += 1
        vmean = float(np.mean(vels))
        rows.append(
            dict(
                target=v,
                score=score,
                length=int(length),
                per_step=score / length,
                vel_mean=vmean,
                vel_err=abs(vmean - v),
                terminal=bool(done),
            )
        )
    return rows


def summarize(rows):
    a = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    return dict(
        target=rows[0]["target"],
        n=len(rows),
        score_mean=float(a("score").mean()),
        score_std=float(a("score").std()),
        per_step_mean=float(a("per_step").mean()),
        length_mean=float(a("length").mean()),
        vel_mean=float(a("vel_mean").mean()),
        vel_err_mean=float(a("vel_err").mean()),
        fall_rate=float((a("length") < 0.9 * 1000).mean()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", default="6000000")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--speeds", type=float, nargs="+", default=DEFAULT_SPEEDS)
    p.add_argument(
        "--out",
        default=None,
        help="json output path (default LEAPS/results/vcond_heldout/<arm>_<ckpt>.json)",
    )
    args = p.parse_args()

    env, agent, cfg = build_env_and_agent(args.run_dir, args.checkpoint)
    arm = cfg["tonic"]["name"].split("/")[-1]

    per_speed, all_rows = [], []
    for v in args.speeds:
        rows = run_speed(env, agent, v, args.episodes)
        all_rows += rows
        s = summarize(rows)
        per_speed.append(s)
        band = "interp" if TRAIN_LO <= v <= TRAIN_HI else "extrap"
        print(
            f"{arm:22s} v={v:.2f} [{band}]  score {s['score_mean']:6.1f}"
            f" ±{s['score_std']:5.1f}  per_step {s['per_step_mean']:.3f}"
            f"  vel {s['vel_mean']:.2f} (err {s['vel_err_mean']:.3f})"
            f"  len {s['length_mean']:.0f}  fall {s['fall_rate']:.0%}"
        )

    out = args.out or os.path.join(
        Path(__file__).resolve().parents[3],
        "results",
        "vcond_heldout",
        f"{arm}_{args.checkpoint}.json",
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(
        dict(arm=arm, run_dir=args.run_dir, checkpoint=args.checkpoint,
             episodes=args.episodes, per_speed=per_speed, rows=all_rows),
        open(out, "w"),
        indent=2,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
