"""Phase-resolved comparison of a trained policy's EMG prior against real EMG.

Every RL-side diagnostic logged so far (latent_residual_share, absmean, ...)
is a scalar averaged over an entire episode -- it can't say whether a_hat
actually bursts at the right point in the gait cycle, only how big it is on
average. This replays a saved checkpoint, detects individual gait cycles
from right-foot contact force, time-normalizes each mapped muscle's a_hat /
residual / final_action to 0-100% of the cycle, and overlays the result
against the real EMG mean profile the decoder was trained on.

Usage:
    python -m leaps.scripts.plot_gait_cycle_rollout \
        --run-dir /media/calc_2/scone_results/live/sconewalk_h0918_latent6_seed0_rerun/260713.232637.H0918v2j \
        --model-name h0918
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from deprl import env_wrappers
from deprl.play import get_paths
from deprl.utils import load_checkpoint
from deprl.vendor.tonic import logger as deprl_logger

from leaps.data.metadata import LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides
from leaps.envs.emg_mapping import MODEL_MAPS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Run directory with config.yaml + checkpoints/.")
    p.add_argument("--checkpoint", default="last")
    p.add_argument("--model-name", required=True, choices=["h0918", "h1622", "h2190"])
    p.add_argument("--num-steps", type=int, default=6000, help="Max sim steps to roll out.")
    p.add_argument("--min-cycles", type=int, default=15, help="Stop once this many valid cycles are collected.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--contact-threshold", type=float, default=2.0, help="calcn_r contact-force magnitude (N) counted as ground contact.")
    p.add_argument("--output-dir", default=None)
    return p.parse_args(argv)


def load_agent_and_env(run_dir, checkpoint, seed):
    """Mirrors deprl.play.play()'s loading logic, but returns objects instead
    of handing off to play_scone -- we need the raw environment/wrapper
    chain to pull per-step diagnostics deprl's own play loop doesn't expose.
    """
    path, checkpoint, checkpoint_path = get_paths(run_dir, checkpoint, None)
    config, checkpoint_path, _ = load_checkpoint(checkpoint_path, checkpoint)

    header = config["tonic"]["header"]
    agent_str = config["tonic"]["agent"]
    env_str = config["tonic"].get("test_environment") or config["tonic"]["environment"]

    if header:
        exec(header, globals())

    agent = eval(agent_str)
    environment = eval(env_str)
    environment.seed(seed)
    environment = env_wrappers.apply_wrapper(environment)
    if "env_args" in config:
        environment.merge_args(config["env_args"])
        environment.apply_args()

    if "mpo_args" in config:
        agent.set_params(**config["mpo_args"])
    agent.initialize(
        observation_space=environment.observation_space,
        action_space=environment.action_space,
        seed=seed,
    )
    agent.load(checkpoint_path, only_checkpoint=True)
    return agent, environment


def find_latent_wrapper(environment):
    """Walk the gym.Wrapper chain to the LatentActionPriorWrapper instance --
    it's a middle layer (SconeWrapper is outermost), so neither `environment`
    itself nor `.unwrapped` reaches it directly.
    """
    w = environment
    seen = 0
    while not hasattr(w, "_mapped_names"):
        if not hasattr(w, "env"):
            raise RuntimeError("LatentActionPriorWrapper not found in the wrapper chain.")
        w = w.env
        seen += 1
        if seen > 20:
            raise RuntimeError("Wrapper chain too deep -- something's wrong.")
    return w


def rollout(agent, environment, latent_wrapper, num_steps, min_cycles, contact_threshold, clip_hi):
    """Steps the env, recording per-step a_hat/final_action (clipped, matching
    what the simulator actually applies) and right-foot contact force for
    gait-cycle segmentation. Resets on fall and keeps going.
    """
    mapped_mask = latent_wrapper._mapped_mask
    mapped_names = latent_wrapper._mapped_names

    a_hat_log, final_log, contact_log = [], [], []

    observations = environment.reset()
    muscle_states = environment.muscle_states
    body_names = [b.name() for b in environment.unwrapped.model.bodies()]
    calcn_r_idx = body_names.index("calcn_r")

    for step in range(num_steps):
        actions = agent.test_step(observations, muscle_states=muscle_states, steps=1e6)
        if len(actions.shape) > 1:
            actions = actions[0, :]
        observations, _, done, _ = environment.step(actions)
        muscle_states = environment.muscle_states

        a_hat = np.clip(latent_wrapper._last_a_hat, 0.0, clip_hi)[mapped_mask]
        final_action = np.clip(latent_wrapper._last_final_action, 0.0, clip_hi)[mapped_mask]
        contact = np.abs(
            environment.unwrapped.model.bodies()[calcn_r_idx].contact_force().array()
        ).sum()

        a_hat_log.append(a_hat)
        final_log.append(final_action)
        contact_log.append(contact)

        if done:
            observations = environment.reset()
            muscle_states = environment.muscle_states

        n_cycles = estimate_cycles(np.array(contact_log), contact_threshold)
        if n_cycles >= min_cycles:
            break

    return (
        np.array(a_hat_log), np.array(final_log), np.array(contact_log), mapped_names
    )


def estimate_cycles(contact, threshold):
    on = contact > threshold
    rising = np.where(on[1:] & ~on[:-1])[0] + 1
    return max(0, len(rising) - 1)


def segment_cycles(contact, threshold, min_len=15, max_len=120):
    on = contact > threshold
    rising = np.where(on[1:] & ~on[:-1])[0] + 1
    cycles = []
    for i in range(len(rising) - 1):
        lo, hi = rising[i], rising[i + 1]
        if min_len <= (hi - lo) <= max_len:
            cycles.append((lo, hi))
    return cycles


def normalize_cycle(trace, lo, hi, n_points=101):
    """Interpolate one cycle's per-step trace to n_points over 0-100%."""
    x_old = np.linspace(0, 100, hi - lo)
    x_new = np.linspace(0, 100, n_points)
    return np.interp(x_new, x_old, trace[lo:hi])


def real_emg_profiles(model_name, n_points=101):
    """Real EMG mean profile per mapped actuator (averaging constituent
    channels the same way emg_mapping.EMGToMuscleMapper does), from the same
    data the decoder was trained on -- not a simulation, ground truth.
    """
    channel_map = MODEL_MAPS[model_name]  # channel -> actuator
    strides, _ = load_strides(LEAPS_H5_PATH, with_metadata=True)  # (n_strides, 101, 11)
    from leaps.data.metadata import EMG_CHANNELS
    mean_profile = strides.mean(axis=0)  # (101, 11)
    std_profile = strides.std(axis=0)

    actuator_to_channels = {}
    for ch, act in channel_map.items():
        actuator_to_channels.setdefault(act, []).append(ch)

    out = {}
    for act, chans in actuator_to_channels.items():
        idxs = [EMG_CHANNELS.index(c) for c in chans]
        out[act] = (mean_profile[:, idxs].mean(axis=1), std_profile[:, idxs].mean(axis=1))
    return out


def main(argv=None):
    args = parse_args(argv)
    deprl_logger.log(f"Loading checkpoint from {args.run_dir}")
    agent, environment = load_agent_and_env(args.run_dir, args.checkpoint, args.seed)
    latent_wrapper = find_latent_wrapper(environment)
    clip_hi = 0.5 if getattr(environment.unwrapped, "clip_actions", True) else 1.0

    deprl_logger.log("Rolling out...")
    a_hat_log, final_log, contact_log, mapped_names = rollout(
        agent, environment, latent_wrapper,
        args.num_steps, args.min_cycles, args.contact_threshold, clip_hi,
    )

    cycles = segment_cycles(contact_log, args.contact_threshold)
    deprl_logger.log(f"Collected {len(contact_log)} steps, {len(cycles)} valid gait cycles.")
    if len(cycles) < 3:
        raise RuntimeError(
            f"Only found {len(cycles)} valid cycles -- contact threshold "
            f"({args.contact_threshold}) or num-steps likely needs adjusting. "
            f"contact_log range: [{contact_log.min():.2f}, {contact_log.max():.2f}]"
        )

    n_muscles = len(mapped_names)
    a_hat_norm = np.zeros((len(cycles), 101, n_muscles))
    final_norm = np.zeros((len(cycles), 101, n_muscles))
    for i, (lo, hi) in enumerate(cycles):
        for m in range(n_muscles):
            a_hat_norm[i, :, m] = normalize_cycle(a_hat_log[:, m], lo, hi)
            final_norm[i, :, m] = normalize_cycle(final_log[:, m], lo, hi)

    real_profiles = real_emg_profiles(args.model_name)
    x = np.linspace(0, 100, 101)

    ncols = 3
    nrows = (n_muscles + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)

    for i, name in enumerate(mapped_names):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        ax2 = ax.twinx()

        if name in real_profiles:
            real_mean, real_std = real_profiles[name]
            ax.plot(x, real_mean, "k-", linewidth=2, label="Real EMG (mean)")
            ax.fill_between(x, real_mean - real_std, real_mean + real_std, color="k", alpha=0.12)
        ax.set_ylabel("Real EMG (a.u.)", fontsize=8, color="k")
        ax.tick_params(axis="y", labelsize=7, labelcolor="k")

        a_hat_mean = a_hat_norm[:, :, i].mean(axis=0)
        a_hat_std = a_hat_norm[:, :, i].std(axis=0)
        final_mean = final_norm[:, :, i].mean(axis=0)

        ax2.plot(x, a_hat_mean, color="#2a78d6", linewidth=2, label="a_hat (EMG prior, sim)")
        ax2.fill_between(x, a_hat_mean - a_hat_std, a_hat_mean + a_hat_std, color="#2a78d6", alpha=0.15)
        ax2.plot(x, final_mean, color="#e34948", linewidth=1.6, linestyle="--", label="final action (prior + residual)")
        ax2.set_ylabel("Sim activation", fontsize=8, color="#2a78d6")
        ax2.tick_params(axis="y", labelsize=7, labelcolor="#2a78d6")

        ax.set_title(name, fontsize=10)
        ax.set_xlabel("% gait cycle", fontsize=8)
        ax.set_xlim(0, 100)

        if i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=8)

    for i in range(n_muscles, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"{Path(args.run_dir).parent.name} -- phase-resolved a_hat vs real EMG "
        f"({len(cycles)} gait cycles)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.run_dir) / "gait_cycle_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "gait_cycle_a_hat_vs_emg.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    deprl_logger.log(f"Saved {save_path}")

    # Cross-correlation lag per muscle -- quantifies phase misalignment
    # instead of relying on eyeballing the plot.
    deprl_logger.log("\nPhase-lag check (positive = a_hat trails real EMG, in % gait cycle):")
    for i, name in enumerate(mapped_names):
        if name not in real_profiles:
            continue
        real_mean, _ = real_profiles[name]
        sim_mean = a_hat_norm[:, :, i].mean(axis=0)
        real_z = (real_mean - real_mean.mean()) / (real_mean.std() + 1e-8)
        sim_z = (sim_mean - sim_mean.mean()) / (sim_mean.std() + 1e-8)
        corr = np.correlate(sim_z, np.tile(real_z, 2), mode="valid")  # circular xcorr
        lag = np.argmax(corr[:101])
        lag_pct = lag if lag <= 50 else lag - 101
        peak_corr = corr[:101].max() / 101
        deprl_logger.log(f"  {name:14s} lag={lag_pct:+4d}%  peak_norm_corr={peak_corr:+.2f}")


if __name__ == "__main__":
    main()
