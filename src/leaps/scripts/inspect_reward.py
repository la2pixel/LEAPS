"""Visualize the HaeufleReward landscape before training.

Plots reward components as a function of velocity and muscle activation
to verify the reward doesn't incentivize bad actions.

Usage:
    python -m leaps.scripts.inspect_reward
    python -m leaps.scripts.inspect_reward --out plots/reward/
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Reward parameters (match conf_ppo_phase2_*.yaml) ──────────────────────────
TARGET_VEL      = 1.25   # m/s
ALIVE_BONUS     = 0.1
MAX_EFFORT_W    = 0.1    # max_effort_weight
EFFORT_RAMP     = 300    # steps to ramp alpha 0 → max_effort_weight
NACTIVE_W       = 0.01   # nactive_weight
NACTIVE_THRESH  = 0.1    # activation threshold for N_active count


# ── Pure-numpy reward components ──────────────────────────────────────────────

def vel_reward(x_vel):
    """Flat at 1.0 above target, Gaussian ramp below."""
    return np.where(x_vel >= TARGET_VEL, 1.0, np.exp(-np.square(x_vel - TARGET_VEL)))


def effort(mean_act):
    """Cubic effort: mean(|a|^3) for uniform activation = mean_act."""
    return mean_act ** 3


def alpha(step):
    """Adaptive effort weight: ramps 0 → MAX_EFFORT_W over EFFORT_RAMP steps."""
    return MAX_EFFORT_W * min(step / EFFORT_RAMP, 1.0)


def n_active(mean_act):
    """Fraction of muscles above threshold. Approximated as 1.0 when mean > threshold."""
    # For a flat activation = mean_act: if mean_act > threshold, all muscles count.
    return np.where(mean_act > NACTIVE_THRESH, 1.0, 0.0)


def total_reward(x_vel, mean_act, step):
    a = alpha(step)
    return (ALIVE_BONUS
            + vel_reward(x_vel)
            - a * effort(mean_act)
            - NACTIVE_W * n_active(mean_act))


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_all(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Velocity reward curve ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    vels = np.linspace(-0.5, 2.5, 400)
    ax.plot(vels, vel_reward(vels), lw=2, label="vel_reward")
    ax.axhline(ALIVE_BONUS, color="gray", ls="--", lw=1, label=f"alive_bonus={ALIVE_BONUS}")
    ax.axvline(TARGET_VEL, color="red", ls=":", lw=1.5, label=f"target={TARGET_VEL} m/s")
    ax.axvline(0.0, color="black", ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("Pelvis forward velocity (m/s)")
    ax.set_ylabel("Reward component")
    ax.set_title("Velocity reward (flat above target, Gaussian below)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "1_vel_reward.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 1_vel_reward.png")

    # ── Figure 2: Effort + N_active penalties vs mean activation ─────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    acts = np.linspace(0.0, 1.0, 300)
    for step, ls in [(0, "--"), (150, "-."), (300, "-")]:
        a = alpha(step)
        label = f"effort penalty  (step={step}, α={a:.3f})"
        ax.plot(acts, a * acts**3, ls=ls, lw=2, label=label)
    ax.plot(acts, NACTIVE_W * np.where(acts > NACTIVE_THRESH, 1.0, 0.0),
            color="orange", lw=1.5, label=f"nactive penalty (w={NACTIVE_W})")
    ax.set_xlabel("Mean muscle activation (uniform assumed)")
    ax.set_ylabel("Penalty magnitude")
    ax.set_title("Effort + N_active penalties vs activation level")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "2_penalties.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 2_penalties.png")

    # ── Figure 3: Total reward heatmap (vel × activation) ────────────────────
    vels_2d  = np.linspace(-0.2, 2.0, 200)
    acts_2d  = np.linspace(0.0, 1.0, 200)
    VV, AA   = np.meshgrid(vels_2d, acts_2d)

    steps_to_show = [0, 150, 300]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, step in zip(axes, steps_to_show):
        R = total_reward(VV, AA, step)
        vmin, vmax = -0.5, 1.2
        im = ax.pcolormesh(VV, AA, R, cmap="RdYlGn", vmin=vmin, vmax=vmax, shading="auto")
        ax.axvline(TARGET_VEL, color="white", ls="--", lw=1.5, label=f"target={TARGET_VEL}")
        ax.axhline(NACTIVE_THRESH, color="cyan", ls=":", lw=1, label=f"nactive thresh")
        ax.set_xlabel("Forward velocity (m/s)")
        if ax is axes[0]:
            ax.set_ylabel("Mean muscle activation")
        a = alpha(step)
        ax.set_title(f"step={step}  α={a:.3f}")
        plt.colorbar(im, ax=ax, label="total reward")

    fig.suptitle("HaeufleReward: total reward = alive + vel − α·effort − nactive_w·N_active",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "3_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 3_heatmap.png")

    # ── Figure 4: Reward at target velocity vs activation (sanity check) ──────
    fig, ax = plt.subplots(figsize=(7, 4))
    acts = np.linspace(0.0, 1.0, 300)
    for step, ls in [(0, "--"), (150, "-."), (300, "-")]:
        r = total_reward(TARGET_VEL, acts, step)
        ax.plot(acts, r, ls=ls, lw=2, label=f"step={step}  α={alpha(step):.3f}")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.4)
    ax.set_xlabel("Mean muscle activation (at v = 1.25 m/s)")
    ax.set_ylabel("Total reward")
    ax.set_title("Reward at target velocity: does activating all muscles still pay off?")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "4_reward_at_target_vel.png", dpi=150)
    plt.close(fig)
    print(f"Saved: 4_reward_at_target_vel.png")

    # ── Print key reward values ───────────────────────────────────────────────
    print("\n── Key reward values ───────────────────────────────────────────")
    print(f"Standing still (v=0, a=0):         {total_reward(0.0,  0.0,  300):.4f}")
    print(f"Standing still (v=0, a=0) step 0:  {total_reward(0.0,  0.0,  0):.4f}")
    print(f"At target speed, zero effort:       {total_reward(1.25, 0.0,  300):.4f}")
    print(f"At target speed, half activation:   {total_reward(1.25, 0.5,  300):.4f}")
    print(f"At target speed, full activation:   {total_reward(1.25, 1.0,  300):.4f}")
    print(f"At target speed, full act, step 0:  {total_reward(1.25, 1.0,  0):.4f}")
    print(f"Backward (v=-0.5, a=0):             {total_reward(-0.5, 0.0,  300):.4f}")
    print(f"Max possible reward (v≥1.25, a=0):  {total_reward(2.0,  0.0,  300):.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="plots/reward/")
    args = parser.parse_args()
    plot_all(Path(args.out))
    print(f"\nAll plots saved to: {args.out}")


if __name__ == "__main__":
    main()
