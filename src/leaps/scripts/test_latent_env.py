"""Smoke test for LatentActionPriorWrapper, same pattern as sconegym/test_environments.py.

Confirms env.unwrapped.model.actuators() is populated at wrap-time (before any
reset()), and that a few random-action steps run without error.

Usage:
    python -m leaps.scripts.test_latent_env \\
        --decoder-path results/synergy/decoder_k8/decoder.pt \\
        --norm-path results/synergy/decoder_k8/norm.npz \\
        --dim-latent 8
"""

from __future__ import annotations

import argparse

import gym
import sconegym  # noqa: F401 -- registers scone* env ids

from leaps.envs import LatentActionPriorWrapper


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-id", default="sconerun_h0918-v1")
    p.add_argument("--model-name", default="h0918")
    p.add_argument("--decoder-path", required=True)
    p.add_argument("--norm-path", required=True)
    p.add_argument("--dim-latent", type=int, required=True)
    p.add_argument("--residual-weight", type=float, default=0.5)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--steps", type=int, default=200)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    base_env = gym.make(args.env_id)
    print(f"gym.make({args.env_id!r}) OK, unwrapped type: {type(base_env.unwrapped)}")

    env = LatentActionPriorWrapper(
        base_env,
        model_name=args.model_name,
        decoder_path=args.decoder_path,
        norm_path=args.norm_path,
        dim_latent=args.dim_latent,
        residual_weight=args.residual_weight,
    )
    print(f"LatentActionPriorWrapper OK. action_space: {env.action_space}")
    print(f"n_actuators: {env.n_actuators}  dim_latent: {env.dim_latent}")

    for ep in range(args.episodes):
        env.reset()
        total_reward = 0.0
        for step in range(1, args.steps + 1):
            action = env.action_space.sample()
            _, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        print(f"Episode {ep}: steps={step} total_reward={total_reward:.3f}")

    env.close()
    print("test_latent_env: OK")


if __name__ == "__main__":
    main()
