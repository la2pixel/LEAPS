"""Extract the raw pre-decode latent (z_u, action[:dim_latent]) a trained
policy actually produces during rollout -- checks whether the decoder's
usable z-range (set by the penalty threshold/scale during AE training,
see notebooks/ab06_decoder_data.ipynb) is something the policy actually
exercises, or mostly unused headroom.

Checkpoint-loading sequence copied from
leaps/models/extract_rollout_activations.py (itself copied from
deprl/play.py::play()) -- same reasoning: existing tested path, not
reinvented, just recording a different per-step quantity.
"""

from __future__ import annotations

import argparse

import numpy as np
from deprl import env_wrappers
from deprl.utils import load_checkpoint


def extract(checkpoint_dir: str, dim_latent: int, n_episodes: int, seed: int = 0) -> np.ndarray:
    if not checkpoint_dir.endswith("/") and not checkpoint_dir.endswith("checkpoints"):
        checkpoint_dir = checkpoint_dir + "/"
    config, checkpoint_path, _ = load_checkpoint(checkpoint_dir, "last")
    if checkpoint_path is None:
        raise FileNotFoundError(f"no checkpoint found under {checkpoint_dir!r}")

    header = config["tonic"]["header"]
    agent_str = config["tonic"]["agent"]
    env_str = config["tonic"]["environment"]

    exec(header)  # noqa: S102

    agent = eval(agent_str)  # noqa: S307
    # deprl/main.py applies config['mpo_args'] (hidden_size, lr_*, etc.) via
    # this exact call after construction -- deprl/play.py's own eval(tonic.
    # agent) reload never does this, so a checkpoint from a non-default net
    # width (e.g. net512) silently fails to load (shape mismatch -> falls
    # back to an untrained model, no error raised). Matched here, not fixed
    # in play.py -- scope this fix to this script only.
    if "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
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

    z_u_trace = []
    obs = environment.reset()
    muscle_states = environment.muscle_states
    length = 0
    episodes = 0
    fell = 0

    while episodes < n_episodes:
        actions = agent.test_step(obs, muscle_states=muscle_states, steps=1e6)
        if len(actions.shape) > 1:
            actions = actions[0, :]
        z_u_trace.append(actions[:dim_latent].copy())
        obs, reward, done, info = environment.step(actions)
        muscle_states = environment.muscle_states
        length += 1

        if done or length >= environment.max_episode_steps:
            episodes += 1
            if done and length < environment.max_episode_steps:
                fell += 1
            obs = environment.reset()
            muscle_states = environment.muscle_states
            length = 0

    print(f"{episodes} episodes collected, {fell} ended in a fall (not a timeout)")
    return np.array(z_u_trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dim-latent", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    z = extract(args.checkpoint_dir, args.dim_latent, args.episodes, args.seed)
    print(f"collected z_u trace shape: {z.shape}")
    print(f"per-dim min:  {z.min(axis=0)}")
    print(f"per-dim max:  {z.max(axis=0)}")
    print(f"per-dim mean: {z.mean(axis=0)}")
    print(f"per-dim std:  {z.std(axis=0)}")
    np.save(args.out, z)
    print(f"saved to {args.out}")
