"""Visualize the EMG processing pipeline for one subject.

Generates 4 diagnostic plots showing each processing stage:
  1. Raw EMG vs rectified envelope for 3 channels
  2. Stride segmentation overlaid on continuous EMG
  3. All time-normalized strides as a heatmap (101 points x 11 channels)
  4. Activation distribution — what the autoencoder will see

Run from cluster:
    python3 scripts/visualize_emg.py
    python3 scripts/visualize_emg.py --subject AB10
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from leaps.data.loaders import discover_trials, load_mat_table
from leaps.data.metadata import EMG_CHANNEL_LABELS
from leaps.data.processing import (
    find_stride_intervals,
    process_subject,
    process_trial_emg,
    rectify_emg,
)

DATA_ROOT = "/fast/lsivakumar/datasets/camargo"
OUT_DIR = Path(__file__).resolve().parent / "plots"


def plot_raw_vs_rectified(emg_table, gc_table, trial_name, out_dir):
    """Plot 1: Raw EMG alongside its rectified envelope for 3 channels."""
    header = emg_table["Header"]
    channels = [k for k in emg_table if k != "Header"]
    raw = np.column_stack([emg_table[ch] for ch in channels])
    rect = rectify_emg(raw, fs=1000)

    show = ["gastrocmed", "rectusfemoris", "tibialisanterior"]
    show_idx = [channels.index(ch) for ch in show if ch in channels]

    fig, axes = plt.subplots(len(show_idx), 1, figsize=(14, 3 * len(show_idx)), sharex=True)
    if len(show_idx) == 1:
        axes = [axes]

    for ax, idx in zip(axes, show_idx):
        ch = channels[idx]
        ax.plot(header, raw[:, idx], linewidth=0.2, alpha=0.5, color="C0", label="Raw")
        ax.plot(header, rect[:, idx], linewidth=1.2, color="C3", label="Envelope")
        ax.set_ylabel(EMG_CHANNEL_LABELS.get(ch, ch), fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.margins(x=0)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Raw vs Rectified EMG — {trial_name}", fontsize=11)
    fig.tight_layout()
    out = out_dir / "01_raw_vs_rectified.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_stride_segmentation(emg_table, gc_table, trial_name, out_dir):
    """Plot 2: Rectified EMG with heel-strike segmentation lines."""
    header = emg_table["Header"]
    channels = [k for k in emg_table if k != "Header"]
    raw = np.column_stack([emg_table[ch] for ch in channels])
    rect = rectify_emg(raw, fs=1000)

    gc_header = gc_table["Header"]
    intervals = find_stride_intervals(gc_header, gc_table["HeelStrike"])

    # Show one channel with stride boundaries
    ch_idx = channels.index("gastrocmed") if "gastrocmed" in channels else 0
    ch_name = EMG_CHANNEL_LABELS.get(channels[ch_idx], channels[ch_idx])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Top: gait cycle sawtooth
    ax1.plot(gc_header, gc_table["HeelStrike"], linewidth=0.8, color="C2")
    ax1.set_ylabel("Gait cycle %")
    ax1.set_title(f"Stride segmentation — {trial_name}")

    # Bottom: rectified EMG with stride boundaries
    ax2.plot(header, rect[:, ch_idx], linewidth=0.6, color="C3")
    for t_start, _ in intervals:
        ax2.axvline(t_start, color="C2", alpha=0.5, linewidth=0.5)
    ax2.set_ylabel(ch_name)
    ax2.set_xlabel("Time (s)")
    ax2.margins(x=0)
    ax2.annotate(f"{len(intervals)} strides detected", xy=(0.02, 0.95),
                 xycoords="axes fraction", fontsize=9, va="top",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.tight_layout()
    out = out_dir / "02_stride_segmentation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_stride_heatmap(strides_3d, out_dir, subject):
    """Plot 3: Heatmap of all time-normalized strides (channels x gait %)."""
    from leaps.data.metadata import EMG_CHANNELS

    # Average across strides for a clean profile
    mean_stride = strides_3d.mean(axis=0)  # (101, 11)
    std_stride = strides_3d.std(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 2]})

    # Top: heatmap of mean stride
    im = axes[0].imshow(
        mean_stride.T, aspect="auto", cmap="hot", interpolation="nearest",
        extent=[0, 100, 0, mean_stride.shape[1]],
    )
    axes[0].set_yticks(np.arange(len(EMG_CHANNELS)) + 0.5)
    axes[0].set_yticklabels([EMG_CHANNEL_LABELS.get(ch, ch) for ch in EMG_CHANNELS], fontsize=7)
    axes[0].set_xlabel("Gait cycle %")
    axes[0].set_title(f"Mean stride profile — {subject} ({strides_3d.shape[0]} strides)")
    plt.colorbar(im, ax=axes[0], label="Normalized activation")

    # Bottom: mean ± std for 3 representative channels
    gait_pct = np.linspace(0, 100, 101)
    show = ["gastrocmed", "rectusfemoris", "tibialisanterior"]
    for ch_name in show:
        if ch_name in EMG_CHANNELS:
            idx = EMG_CHANNELS.index(ch_name)
            label = EMG_CHANNEL_LABELS.get(ch_name, ch_name)
            axes[1].plot(gait_pct, mean_stride[:, idx], linewidth=1.5, label=label)
            axes[1].fill_between(
                gait_pct,
                mean_stride[:, idx] - std_stride[:, idx],
                mean_stride[:, idx] + std_stride[:, idx],
                alpha=0.2,
            )
    axes[1].set_xlabel("Gait cycle %")
    axes[1].set_ylabel("Normalized EMG")
    axes[1].legend(fontsize=8)
    axes[1].set_xlim(0, 100)

    fig.tight_layout()
    out = out_dir / "03_stride_profiles.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_activation_distribution(activations, out_dir, subject):
    """Plot 4: Distribution of activation values — the autoencoder's input."""
    from leaps.data.metadata import EMG_CHANNELS

    fig, axes = plt.subplots(3, 4, figsize=(14, 8))
    axes = axes.flat

    for i, ch in enumerate(EMG_CHANNELS):
        ax = axes[i]
        vals = activations[:, i]
        ax.hist(vals, bins=80, color="C0", alpha=0.7, density=True)
        ax.set_title(EMG_CHANNEL_LABELS.get(ch, ch), fontsize=8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(1, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlim(-0.5, 3.0)
        ax.tick_params(labelsize=7)

    # Use last subplot for summary stats
    ax = axes[11]
    ax.axis("off")
    stats_text = (
        f"Subject: {subject}\n"
        f"Snapshots: {activations.shape[0]:,}\n"
        f"Channels: {activations.shape[1]}\n"
        f"Mean: {activations.mean():.3f}\n"
        f"Std: {activations.std():.3f}\n"
        f"Min: {activations.min():.3f}\n"
        f"Max: {activations.max():.3f}\n"
        f"% in [0,1]: {(((activations >= 0) & (activations <= 1)).mean() * 100):.1f}%"
    )
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="center", fontfamily="monospace")

    fig.suptitle(f"Activation distributions (autoencoder input) — {subject}", fontsize=11)
    fig.tight_layout()
    out = out_dir / "04_activation_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Visualize EMG processing pipeline.")
    parser.add_argument("--subject", default="AB09", help="Subject ID (default: AB09)")
    parser.add_argument("--data-root", default=DATA_ROOT)
    args = parser.parse_args()

    out_dir = OUT_DIR
    out_dir.mkdir(exist_ok=True)
    subject = args.subject

    print(f"Processing {subject} from {args.data_root}")
    trials = discover_trials(args.data_root, subjects=[subject], modes=["treadmill"],
                             sensors=["emg", "gcRight", "conditions"])

    if subject not in trials:
        print(f"ERROR: {subject} not found at {args.data_root}")
        sys.exit(1)

    subj = trials[subject]["treadmill"]
    emg_files = sorted(subj["emg"])
    gc_files = sorted(subj["gcRight"])
    cond_files = sorted(subj["conditions"])

    # Load all trials
    emg_tables = [load_mat_table(f) for f in emg_files]
    gc_tables = [load_mat_table(f) for f in gc_files]
    cond_tables = [load_mat_table(f) for f in cond_files]

    # Plot 1 & 2: use first trial
    trial_name = f"{subject} / {Path(emg_files[0]).stem}"
    print(f"\nPlot 1: raw vs rectified")
    plot_raw_vs_rectified(emg_tables[0], gc_tables[0], trial_name, out_dir)
    print(f"Plot 2: stride segmentation")
    plot_stride_segmentation(emg_tables[0], gc_tables[0], trial_name, out_dir)

    # Run full pipeline for plots 3 & 4
    print(f"Running full pipeline for {subject}...")
    activations, strides_3d, norm_params = process_subject(
        emg_tables, gc_tables, cond_tables,
    )
    print(f"  {strides_3d.shape[0]} strides, {activations.shape[0]} activation snapshots")

    print(f"Plot 3: stride profiles")
    plot_stride_heatmap(strides_3d, out_dir, subject)
    print(f"Plot 4: activation distributions")
    plot_activation_distribution(activations, out_dir, subject)

    print(f"\nAll plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
