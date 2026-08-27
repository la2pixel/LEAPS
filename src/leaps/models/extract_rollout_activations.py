"""Extract raw muscle-activation traces from a trained checkpoint's greedy
rollout -- the data source for the SAR-style self-derived synergy control
(Berg, Caggiano, Kumar 2023, arXiv:2307.03716): fit a synergy basis on a
policy's OWN experience instead of external EMG, to isolate whether the
value of a synergy prior comes from having ANY self-consistent low-dim
structure, or specifically from real biological content.

Checkpoint-loading sequence is copied from deprl/play.py::play() (not
reinvented) -- that function is the existing, tested path for reloading a
trained agent+env from a config.yaml + checkpoint, it just doesn't expose a
hook for recording raw activations per step (it either writes .sto files
via play_scone(), or renders via play_control_suite() -- neither returns
the data this needs). This script does the same setup, then a plain
rollout loop instead.
"""

from __future__ import annotations

import argparse

import numpy as np
from deprl import env_wrappers
from deprl.utils import load_checkpoint


def extract(checkpoint_dir: str, n_episodes: int, seed: int = 0) -> np.ndarray:
    # load_checkpoint() string-concatenates "checkpoints" onto this path with
    # no separator if it doesn't already end in one/"checkpoints" -- a quirk
    # of the existing deprl utility, not something to fix there.
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

        # Same episode-boundary logic as play_scone() -- gaitgym.py's own
        # `done` only fires on falling, never on reaching the horizon (see
        # leaps/envs/adaptive_effort_reward.py's docstring for the full
        # story on why this distinction matters).
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
