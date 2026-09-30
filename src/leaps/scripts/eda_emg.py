"""EMG preprocessing EDA — visual walkthrough of the full pipeline.
THIS SHOULD BE A NOTEBOOK

Generates 6 figures saved as PNGs:
  fig1_pipeline.png       — filter chain step-by-step for one trial (one muscle)
  fig2_segmentation.png   — HeelStrike sawtooth + stride boundary overlay
  fig3_strides.png        — time-normalized spaghetti plots for all 11 muscles
  fig4_normalization.png  — inter-subject amplitude before vs after normalization
  fig5_heatmap.png        — full N_strides × 101 heatmap per muscle
  fig6_conditions.png     — mean stride profile by locomotion mode per muscle

Figures 1–2 require raw .mat files (--data-root).
Figures 3–6 require the processed HDF5 (--h5).

Usage:
    python -m leaps.scripts.eda_emg \\
        --data-root $CAMARGO_DATA \\
        --h5 $LEAPS_EMG_H5 \\
        --subject AB09 \\
        --output-dir outputs/eda
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, firwin

from leaps.data.metadata import EMG_CHANNELS, EMG_CHANNEL_LABELS
from leaps.data import (
    load_mat_table, discover_trials,
    find_stride_intervals, segment_strides, time_normalize_stride,
    compute_normalization,
)

# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

MUSCLE_LABELS = [EMG_CHANNEL_LABELS[ch] for ch in EMG_CHANNELS]
CONDITION_COLORS = {
    "treadmill":   "#2196F3",
    "levelground": "#4CAF50",
    "ramp":        "#FF9800",
    "stair":       "#9C27B0",
}


# ── Figure 1: filter chain pipeline ───────────────────────────────────────────

def fig_pipeline(emg_raw: np.ndarray, gc_table: dict, fs: int,
                 channel: int, t_window: tuple, out_path: Path) -> None:
    """Five-row trace showing each filtering step for one muscle."""

    ch_name = MUSCLE_LABELS[channel]
    t_start, t_end = t_window
    mask = (emg_raw[:, 0] >= t_start) & (emg_raw[:, 0] < t_end)
    t = emg_raw[mask, 0]
    raw = emg_raw[mask, channel + 1]   # +1 because col 0 is header

    # Replicate the filter chain step by step to capture intermediates
    x = raw.astype(np.float64)

    b, a = butter(4, 10.0, btype="high", fs=fs)
    after_hp = filtfilt(b, a, x)

    b_fir = firwin(21, [10.0, 450.0], pass_zero=False, fs=fs)
    after_bp = filtfilt(b_fir, [1.0], after_hp)
    after_rect = np.abs(after_bp)

    b, a = butter(4, 6.0, btype="low", fs=fs)
    envelope = np.abs(filtfilt(b, a, after_rect))

    # Heel strikes for overlay on the last panel
    gc_mask = (gc_table["Header"] >= t_start) & (gc_table["Header"] < t_end)
    gc_t = gc_table["Header"][gc_mask]
    gc_hs = gc_table["HeelStrike"][gc_mask]
    stride_intervals = find_stride_intervals(gc_table["Header"], gc_table["HeelStrike"])
    hs_times = [iv[0] for iv in stride_intervals
                if t_start <= iv[0] < t_end]

    steps = [
        (raw,        "#555555", "Raw EMG (mV)",          None),
        (after_hp,   "#1565C0", "After highpass 10 Hz",  None),
        (after_bp,   "#6A1B9A", "After bandpass 10–450 Hz", None),
        (after_rect, "#C62828", "Full-wave rectified",   None),
        (envelope,   "#2E7D32", "Envelope (lowpass 6 Hz)", hs_times),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"EMG Filter Chain — {ch_name}", fontsize=11, fontweight="bold")

    for ax, (signal, color, label, hs) in zip(axes, steps):
        ax.plot(t, signal, color=color, lw=0.7, rasterized=True)
        ax.set_ylabel(label, fontsize=7.5)
        ax.axhline(0, color="black", lw=0.4, ls="--", alpha=0.4)
        if hs:
            for hs_t in hs:
                ax.axvline(hs_t, color="#FF6F00", lw=1.0, ls="--", alpha=0.7)
        ax.margins(x=0)

    axes[-1].set_xlabel("Time (s)")

    # Inset: HeelStrike sawtooth
    ax_gc = axes[-1].inset_axes([0.0, -0.45, 1.0, 0.35])
    ax_gc.plot(gc_t, gc_hs, color="#FF6F00", lw=0.8)
    ax_gc.set_ylabel("Gait %", fontsize=7)
    ax_gc.set_xlabel("Time (s)", fontsize=7)
    ax_gc.set_ylim(-5, 110)
    ax_gc.spines["top"].set_visible(False)
    ax_gc.spines["right"].set_visible(False)
    for hs_t in hs_times:
        ax_gc.axvline(hs_t, color="#FF6F00", lw=1.0, ls="--", alpha=0.7)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 2: stride segmentation ─────────────────────────────────────────────

def fig_segmentation(emg_raw: np.ndarray, gc_table: dict, fs: int,
                     channel: int, t_window: tuple, out_path: Path) -> None:
    """HeelStrike sawtooth + envelope with alternating stride backgrounds."""

    t_start, t_end = t_window
    ch_name = MUSCLE_LABELS[channel]

    # Compute envelope
    t_all = emg_raw[:, 0]
    raw_all = emg_raw[:, channel + 1].astype(np.float64)
    b, a = butter(4, 10.0, btype="high", fs=fs)
    x = filtfilt(b, a, raw_all)
    b_fir = firwin(21, [10.0, 450.0], pass_zero=False, fs=fs)
    x = filtfilt(b_fir, [1.0], x)
    x = np.abs(x)
    b, a = butter(4, 6.0, btype="low", fs=fs)
    envelope_all = np.abs(filtfilt(b, a, x))

    mask = (t_all >= t_start) & (t_all < t_end)
    t = t_all[mask]
    envelope = envelope_all[mask]

    gc_mask = (gc_table["Header"] >= t_start) & (gc_table["Header"] < t_end)
    gc_t = gc_table["Header"][gc_mask]
    gc_hs = gc_table["HeelStrike"][gc_mask]

    intervals = find_stride_intervals(gc_table["Header"], gc_table["HeelStrike"])
    intervals_win = [(s, e) for s, e in intervals if t_start <= s < t_end and e <= t_end + 0.5]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    fig.suptitle("Stride Segmentation via Right HeelStrike", fontsize=11, fontweight="bold")

    # Panel 1: sawtooth
    ax1.plot(gc_t, gc_hs, color="#FF6F00", lw=1.0)
    ax1.set_ylabel("Gait cycle %")
    ax1.set_ylim(-5, 110)
    for s, e in intervals_win:
        ax1.axvline(s, color="steelblue", lw=1.0, ls="--", alpha=0.8)

    # Panel 2: envelope with alternating shaded strides
    colors_alt = ["#E3F2FD", "#FFF3E0"]
    for idx, (s, e) in enumerate(intervals_win):
        ax2.axvspan(s, e, color=colors_alt[idx % 2], alpha=0.6, zorder=0)
        ax2.axvline(s, color="steelblue", lw=1.0, ls="--", alpha=0.8, zorder=1)
    ax2.plot(t, envelope, color="#2E7D32", lw=0.9, zorder=2)
    ax2.set_ylabel(f"Envelope — {ch_name} (mV)")
    ax2.set_xlabel("Time (s)")
    ax2.margins(x=0)
    ax1.margins(x=0)

    # Annotate one stride
    if len(intervals_win) >= 2:
        s, e = intervals_win[1]
        mid = (s + e) / 2
        ax2.annotate("1 stride", xy=(mid, envelope.max() * 0.85),
                     ha="center", fontsize=8, color="steelblue",
                     arrowprops=dict(arrowstyle="-|>", color="steelblue"),
                     xytext=(mid, envelope.max() * 1.05))

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 3: time-normalized spaghetti plots ─────────────────────────────────

def fig_strides(strides: np.ndarray, speeds: np.ndarray, out_path: Path,
                speed_ref: float = 1.35) -> None:
    """All strides overlaid per muscle, colored by speed, mean ± std shown."""

    ref_mask = np.abs(speeds - speed_ref) < 0.08
    strides_ref = strides[ref_mask]              # (n_ref, 101, 11)
    all_strides = strides                        # (N, 101, 11)

    gait_pct = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    fig.suptitle(f"Time-normalized EMG Strides — treadmill (~{speed_ref} m/s, n={len(strides_ref)} strides)",
                 fontsize=11, fontweight="bold")

    for ch, ax in enumerate(axes[:11]):
        muscle = strides_ref[:, :, ch]
        # Plot individual strides
        for stride in muscle:
            ax.plot(gait_pct, stride, color="#90CAF9", lw=0.4, alpha=0.4, rasterized=True)
        # Mean ± std
        mean = muscle.mean(axis=0)
        std = muscle.std(axis=0)
        ax.fill_between(gait_pct, mean - std, mean + std, color="#1565C0", alpha=0.25)
        ax.plot(gait_pct, mean, color="#1565C0", lw=1.5)
        ax.set_title(MUSCLE_LABELS[ch], fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(0, 100)
        if ch >= 8:
            ax.set_xlabel("Gait cycle (%)")
        if ch % 4 == 0:
            ax.set_ylabel("Activation [0,1]")

    axes[11].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 4: inter-subject normalization effect ───────────────────────────────

def fig_normalization(h5_path: Path, subjects: list[str], out_path: Path,
                      speed_ref: float = 1.35) -> None:
    """Show raw amplitude spread across subjects, then after normalization."""

    show_channels = [0, 1, 2, 3, 5, 6]   # gastroc, tib, soleus, vastmed, rectfem, bicfem
    gait_pct = np.linspace(0, 100, 101)

    # Collect per-subject mean stride profiles (pre-norm = normalized by their own norms,
    # but we want to show the raw scale effect: multiply back by (max-min)+min per subject)
    raw_means = {ch: [] for ch in show_channels}   # before normalization (a.u. but comparable)
    norm_means = {ch: [] for ch in show_channels}  # after normalization [0,1]

    with h5py.File(h5_path, "r") as hf:
        for subj in subjects:
            if subj not in hf:
                continue
            s_strides = hf[subj]["strides"][:]      # (n, 101, 11) already normalized
            s_speeds = hf[subj]["speeds"][:]
            s_min = hf[subj]["min_vals"][:]
            s_max = hf[subj]["max_vals"][:]

            ref_mask = np.abs(s_speeds - speed_ref) < 0.08
            if ref_mask.sum() < 3:
                continue

            ref_strides = s_strides[ref_mask]       # (n_ref, 101, 11)
            mean_norm = ref_strides.mean(axis=0)    # (101, 11)

            # Undo normalization to get "raw" amplitude (mV envelope units)
            mean_raw = mean_norm * (s_max - s_min) + s_min

            for ch in show_channels:
                raw_means[ch].append(mean_raw[:, ch])
                norm_means[ch].append(mean_norm[:, ch])

    n_ch = len(show_channels)
    fig, axes = plt.subplots(2, n_ch, figsize=(14, 5), sharey="row")
    fig.suptitle("Inter-subject Variability: Before vs After Normalization\n"
                 "(treadmill ~1.35 m/s, each line = one subject)",
                 fontsize=11, fontweight="bold")

    palette = plt.cm.tab20.colors
    row_labels = ["Before norm.\n(mV envelope)", "After norm.\n[0, 1]"]

    for col_idx, ch in enumerate(show_channels):
        for row, means in enumerate([raw_means[ch], norm_means[ch]]):
            ax = axes[row, col_idx]
            for i, m in enumerate(means):
                ax.plot(gait_pct, m, color=palette[i % len(palette)], lw=0.9, alpha=0.7)
            ax.set_title(MUSCLE_LABELS[ch], fontsize=8)
            ax.set_xlim(0, 100)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row], fontsize=7.5)
            if row == 1:
                ax.set_xlabel("Gait %", fontsize=7.5)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 5: full dataset heatmap ────────────────────────────────────────────

def fig_heatmap(strides: np.ndarray, speeds: np.ndarray, modes: np.ndarray,
                out_path: Path, max_strides: int = 400) -> None:
    """N_strides × 101 heatmap for each muscle. Rows sorted by speed."""

    # Sort strides by speed (NaN = non-treadmill → put at end)
    speeds_sort = np.where(np.isnan(speeds), 99.0, speeds)
    order = np.argsort(speeds_sort)
    strides_s = strides[order]
    speeds_s = speeds[order]

    # Subsample if too many strides
    if len(strides_s) > max_strides:
        idx = np.linspace(0, len(strides_s) - 1, max_strides, dtype=int)
        strides_s = strides_s[idx]
        speeds_s = speeds_s[idx]

    gait_pct = np.linspace(0, 100, 101)
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    fig.suptitle(f"Dataset Heatmap: {len(strides_s)} strides × 101 gait% × 11 muscles\n"
                 "(rows sorted by speed: treadmill slow → fast → non-treadmill)",
                 fontsize=10, fontweight="bold")

    for ch, ax in enumerate(axes[:11]):
        img = strides_s[:, :, ch]   # (n_strides, 101)
        im = ax.imshow(img, aspect="auto", origin="upper", vmin=0, vmax=1,
                       cmap="hot", extent=[0, 100, len(strides_s), 0],
                       rasterized=True)
        ax.set_title(MUSCLE_LABELS[ch], fontsize=8)
        ax.set_xlim(0, 100)
        if ch >= 8:
            ax.set_xlabel("Gait cycle (%)", fontsize=7.5)
        if ch % 4 == 0:
            ax.set_ylabel("Stride index\n(sorted by speed)", fontsize=7.5)

    # Shared colorbar
    fig.colorbar(im, ax=axes[:11].tolist(), label="Activation [0,1]",
                 fraction=0.015, pad=0.04)
    axes[11].axis("off")

    # Speed axis annotation on the right
    if not np.all(np.isnan(speeds_s)):
        valid = ~np.isnan(speeds_s)
        n_valid = valid.sum()
        ax_spd = fig.add_axes([0.92, 0.1, 0.01, 0.8])
        ax_spd.set_ylim(len(strides_s), 0)
        ax_spd.set_yticks([0, n_valid // 2, n_valid, len(strides_s)])
        ax_spd.set_yticklabels(
            [f"{speeds_s[0]:.1f}", f"{speeds_s[n_valid//2]:.1f}",
             "non-TM →", ""], fontsize=6)
        ax_spd.set_xticks([])
        ax_spd.set_ylabel("Speed (m/s)", fontsize=6)
        ax_spd.yaxis.set_label_position("right")
        ax_spd.yaxis.tick_right()

    plt.tight_layout(rect=[0, 0, 0.91, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Figure 6: cross-condition mean profiles ────────────────────────────────────

def fig_conditions(strides: np.ndarray, modes: np.ndarray, conditions: np.ndarray,
                   speeds: np.ndarray, out_path: Path) -> None:
    """Mean ± std per locomotion mode for all 11 muscles."""

    gait_pct = np.linspace(0, 100, 101)

    # Define which strides to show per condition
    subsets = {
        "treadmill ~1.25 m/s": (modes == "treadmill") & (np.abs(speeds - 1.25) < 0.08),
        "treadmill ~1.85 m/s": (modes == "treadmill") & (np.abs(speeds - 1.85) < 0.08),
        "levelground normal":  (modes == "levelground") & (conditions == "normal"),
        "ramp 5.2°":           (modes == "ramp") & (conditions == "5.2"),
        "stair 102mm":         (modes == "stair") & (conditions == "102"),
    }
    colors = ["#1565C0", "#0097A7", "#2E7D32", "#FF6F00", "#6A1B9A"]

    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.flatten()
    fig.suptitle("Mean EMG Stride Profile by Locomotion Mode", fontsize=11, fontweight="bold")

    for ch, ax in enumerate(axes[:11]):
        for (label, mask), color in zip(subsets.items(), colors):
            n = mask.sum()
            if n < 3:
                continue
            sub = strides[mask, :, ch]
            mean = sub.mean(axis=0)
            std = sub.std(axis=0)
            ax.fill_between(gait_pct, mean - std, mean + std, color=color, alpha=0.15)
            ax.plot(gait_pct, mean, color=color, lw=1.2, label=f"{label} (n={n})")
        ax.set_title(MUSCLE_LABELS[ch], fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(0, 100)
        if ch >= 8:
            ax.set_xlabel("Gait cycle (%)")
        if ch % 4 == 0:
            ax.set_ylabel("Activation [0,1]")

    # Single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.01),
               fontsize=7.5, framealpha=0.9, ncol=1)

    axes[11].axis("off")
    plt.tight_layout(rect=[0, 0.0, 1, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EMG preprocessing EDA plots.")
    parser.add_argument("--data-root", default=None,
                        help="Raw Camargo dataset root (required for fig1+2)")
    parser.add_argument("--h5", default=None,
                        help="Processed HDF5 path (required for fig3–6)")
    parser.add_argument("--subject", default="AB09",
                        help="Subject to use for per-subject plots (default: AB09)")
    parser.add_argument("--output-dir", default="outputs/eda",
                        help="Directory for output PNGs")
    parser.add_argument("--pipeline-channel", type=int, default=0,
                        help="EMG channel index for fig1+2 (default: 0 = gastrocmed)")
    parser.add_argument("--t-start", type=float, default=5.0,
                        help="Start time (s) for pipeline window (default: 5)")
    parser.add_argument("--t-end", type=float, default=20.0,
                        help="End time (s) for pipeline window (default: 20)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figures 1 & 2: raw pipeline (needs .mat files) ─────────────────────────
    if args.data_root:
        print(f"\n── Loading raw data for {args.subject} ──")
        trials = discover_trials(
            args.data_root, subjects=[args.subject],
            modes=["treadmill"], sensors=["emg", "gcRight"],
        )
        if args.subject not in trials or "treadmill" not in trials[args.subject]:
            print(f"No treadmill data for {args.subject}, skipping fig1+2", file=sys.stderr)
        else:
            emg_files = sorted(trials[args.subject]["treadmill"].get("emg", []))
            gc_files  = sorted(trials[args.subject]["treadmill"].get("gcRight", []))

            if not emg_files or not gc_files:
                print("Missing emg or gcRight files, skipping fig1+2", file=sys.stderr)
            else:
                # Pick the middle trial file (usually steady-state)
                trial_idx = len(emg_files) // 2
                print(f"  Using: {Path(emg_files[trial_idx]).name}")

                emg_table = load_mat_table(emg_files[trial_idx])
                gc_table  = load_mat_table(gc_files[trial_idx])

                if not emg_table or not gc_table:
                    print("Failed to load trial files", file=sys.stderr)
                else:
                    # Build (n_samples, 1+11) array: [Header, ch0, ..., ch10]
                    header = emg_table["Header"]
                    channels = [emg_table[ch] for ch in EMG_CHANNELS if ch in emg_table]
                    emg_raw = np.column_stack([header] + channels)

                    t_window = (args.t_start, args.t_end)

                    print("  Generating fig1_pipeline.png ...")
                    fig_pipeline(emg_raw, gc_table, fs=1000,
                                 channel=args.pipeline_channel,
                                 t_window=t_window,
                                 out_path=out_dir / "fig1_pipeline.png")

                    print("  Generating fig2_segmentation.png ...")
                    fig_segmentation(emg_raw, gc_table, fs=1000,
                                     channel=args.pipeline_channel,
                                     t_window=t_window,
                                     out_path=out_dir / "fig2_segmentation.png")
    else:
        print("--data-root not provided, skipping fig1+2")

    # ── Figures 3–6: dataset-level (needs HDF5) ────────────────────────────────
    if args.h5:
        print(f"\n── Loading HDF5: {args.h5} ──")
        h5_path = Path(args.h5)

        # Per-subject data for fig3
        with h5py.File(h5_path, "r") as hf:
            if args.subject not in hf:
                print(f"Subject {args.subject} not in HDF5", file=sys.stderr)
                subj_strides = None
            else:
                subj_strides  = hf[args.subject]["strides"][:]   # (n, 101, 11)
                subj_speeds   = hf[args.subject]["speeds"][:]
                subj_modes    = np.array(hf[args.subject]["modes"]).astype(str)

        if subj_strides is not None:
            print(f"  {args.subject}: {subj_strides.shape[0]} strides")
            print("  Generating fig3_strides.png ...")
            fig_strides(subj_strides, subj_speeds,
                        out_path=out_dir / "fig3_strides.png")

        # All subjects for fig4
        with h5py.File(h5_path, "r") as hf:
            all_subjects = [k for k in hf.keys()
                            if k not in ("activations", "stride_modes", "stride_subjects",
                                         "stride_speeds", "stride_conditions")]

        print(f"  Found {len(all_subjects)} subjects in HDF5")
        print("  Generating fig4_normalization.png ...")
        fig_normalization(h5_path, all_subjects, out_path=out_dir / "fig4_normalization.png")

        # Pool all subjects for fig5+6
        print("  Pooling all subjects for heatmap + conditions plots ...")
        all_strides_list, all_speeds_list, all_modes_list, all_conds_list = [], [], [], []
        with h5py.File(h5_path, "r") as hf:
            for subj in all_subjects:
                if subj not in hf:
                    continue
                all_strides_list.append(hf[subj]["strides"][:])
                all_speeds_list.append(hf[subj]["speeds"][:])
                all_modes_list.append(np.array(hf[subj]["modes"]).astype(str))
                # conditions may be stored as bytes in older H5 files
                raw_conds = np.array(hf[subj]["conditions"])
                if raw_conds.dtype.kind == "S":
                    raw_conds = raw_conds.astype(str)
                else:
                    raw_conds = raw_conds.astype(str)
                all_conds_list.append(raw_conds)

        all_strides  = np.concatenate(all_strides_list, axis=0)
        all_speeds   = np.concatenate(all_speeds_list, axis=0)
        all_modes    = np.concatenate(all_modes_list, axis=0)
        all_conds    = np.concatenate(all_conds_list, axis=0)
        print(f"  Total: {all_strides.shape[0]} strides across {len(all_subjects)} subjects")

        print("  Generating fig5_heatmap.png ...")
        fig_heatmap(all_strides, all_speeds, all_modes,
                    out_path=out_dir / "fig5_heatmap.png")

        print("  Generating fig6_conditions.png ...")
        fig_conditions(all_strides, all_modes, all_conds, all_speeds,
                       out_path=out_dir / "fig6_conditions.png")
    else:
        print("--h5 not provided, skipping fig3–6")

    print(f"\nAll done. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
