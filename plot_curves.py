"""Plot PPO learning curves: direct vs latent."""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
for variant, color, label in [
    ("ppo_direct", "red", "Direct (18-dim)"),
    ("ppo_latent", "blue", "Latent (9-dim)"),
]:
    tb_dir = f"experiments/{variant}/tb_logs/"
    subdir = [
        os.path.join(tb_dir, d)
        for d in os.listdir(tb_dir)
        if os.path.isdir(os.path.join(tb_dir, d))
    ][0]
    ea = EventAccumulator(subdir)
    ea.Reload()
    for tag, ax, ylabel in [
        ("rollout/ep_rew_mean", ax1, "Mean Episode Reward"),
        ("rollout/ep_len_mean", ax2, "Mean Episode Length (steps)"),
    ]:
        events = ea.Scalars(tag)
        ax.plot(
            [e.step for e in events],
            [e.value for e in events],
            color=color,
            label=label,
        )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training Steps")
        ax.legend()

ax1.set_title("PPO Training: Episode Reward")
ax2.set_title("PPO Training: Episode Length")
fig.suptitle("Latent Action Prior vs Direct Muscle Control (500k steps)", fontsize=13)
plt.tight_layout()
plt.savefig("results/ppo_learning_curves.png", dpi=150)
print("Saved results/ppo_learning_curves.png")
