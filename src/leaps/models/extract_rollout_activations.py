"""Muscle-activation traces from a trained checkpoint's greedy rollout, for
fitting a synergy basis on the policy's own activations (as in SAR, Berg et
al. 2023, arXiv:2307.03716).

Checkpoint loading follows deprl/play.py::play().
"""

from __future__ import annotations

import argparse

import numpy as np
from deprl import env_wrappers
from deprl.utils import load_checkpoint


def extract(checkpoint_dir: str, n_episodes: int, seed: int = 0) -> np.ndarray:
    # load_checkpoint() appends "checkpoints" without a separator
    if not checkpoint_dir.endswith("/") and not checkpoint_dir.endswith("checkpoints"):
        checkpoint_dir = checkpoint_dir + "/"
    config, checkpoint_path, _ = load_checkpoint(checkpoint_dir, "last")
    if checkpoint_path is None:
        raise FileNotFoundError(f"no checkpoint found under {checkpoint_dir!r}")

    header = config["tonic"]["header"]
    agent_str = config["tonic"]["agent"]
    env_str = config["tonic"]["environment"]

    exec(header)  # noqa: S102 -- same pattern as deprl/play.py::play(), imports deprl/gym/sconegym etc.

    agent = eval(agent_str)  # noqa: S307
    environment = eval(env_str)  # noqa: S307
    environment.seed(seed)
    environment = env_wrappers.apply_wrapper(environment)
    if "env_args" in config:
        environment.merge_args(config["env_args"])
        environment.apply_args()

    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )
    agent.load(checkpoint_path, only_checkpoint=True)

    activations = []
    obs = environment.reset()
    muscle_states = environment.muscle_states
    length = 0
    episodes = 0
    fell = 0

    while episodes < n_episodes:
        actions = agent.test_step(obs, muscle_states=muscle_states, steps=1e6)
        if len(actions.shape) > 1:
            actions = actions[0, :]
        obs, reward, done, info = environment.step(actions)
        muscle_states = environment.muscle_states
        activations.append(environment.unwrapped.model.muscle_activation_array().copy())
        length += 1

        # done only fires on falling, not at the horizon
        if done or length >= environment.max_episode_steps:
            episodes += 1
            if done and length < environment.max_episode_steps:
                fell += 1
            obs = environment.reset()
            muscle_states = environment.muscle_states
            length = 0

    print(f"{episodes} episodes collected, {fell} ended in a fall (not a timeout)")
    return np.array(activations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    acts = extract(args.checkpoint_dir, args.episodes, args.seed)
    print(f"collected activations shape: {acts.shape}")
    np.save(args.out, acts)
    print(f"saved to {args.out}")
