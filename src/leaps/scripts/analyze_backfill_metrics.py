"""Analyze CSVs produced by backfill_rl_checkpoint_metrics.py.

Two comparisons, both reported as Spearman rank correlation:

  channel-vs-real: per-EMG-channel realized magnitude (kind=decoded_emg) vs.
    the real dataset's own per-channel mean activity. Real-data reference is
    computed fresh from the h5 dataset every run (leaps.data.stride_dataset),
    not a copied-in constant, so it can't drift out of sync with the data.
    With --checkpoint-sweep, reports one rho per checkpoint_step instead of
    pooling them (for a --sweep-checkpoints backfill CSV).

  real-vs-null: per-actuator magnitude for a given `kind` (e.g.
    residual_unmapped, prior_abs_mapped), compared between run labels starting
    with "real" vs "null". Requires the backfill CSV to have been produced
    with explicit --run-label real_seedN / null_seedN (default run labels
    collide between emg_lap and no_lap configs of the same body/k/w/seed).

coverage-ratio: does the h0918-only magnitude-inversion track how much of
    each body's action space the EMG prior actually covers? For each body,
    builds its env fresh (leaps.scripts.backfill_rl_checkpoint_metrics.build_env
    -- no agent/checkpoint needed, just the LatentActionPriorWrapper's own
    n_actuators/_mapped_names) to get mapped/total actuator coverage, and
    joins it against |rho| pooled from --csv. n=3 bodies -- report the table,
    don't over-read the correlation stat.

Usage:
    python -m leaps.scripts.analyze_backfill_metrics channel-vs-real \\
        --csv LEAPS/results/backfill/h1622_h2190_hardconstraint11_generalization.csv

    python -m leaps.scripts.analyze_backfill_metrics channel-vs-real \\
        --csv LEAPS/results/backfill/h0918_hardconstraint11_temporal_sweep.csv \\
        --checkpoint-sweep

    python -m leaps.scripts.analyze_backfill_metrics real-vs-null \\
        --csv LEAPS/results/backfill/h0918_hardconstraint11_real_vs_null.csv \\
        --kind residual_unmapped

    python -m leaps.scripts.analyze_backfill_metrics coverage-ratio \\
        --csv LEAPS/results/backfill/all_bodies_hardconstraint11_channel_backfill.csv
"""
import argparse
import os

os.environ.setdefault("LEAPS_EMG_H5", "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import pandas as pd
import scipy.stats as st


def real_channel_reference():
    """Mean per-channel EMG activity from the real dataset, e.g. what
    channel-backfill-h0918's ρ=0.927 offline check used as ground truth."""
    from leaps.data.metadata import ALL_MODES, EMG_CHANNELS, LEAPS_H5_PATH
    from leaps.data.stride_dataset import load_strides

    strides, _ = load_strides(LEAPS_H5_PATH, modes=ALL_MODES, with_metadata=True)
    mean_act = strides.reshape(-1, len(EMG_CHANNELS)).mean(axis=0)
    return dict(zip(EMG_CHANNELS, mean_act))


def _channel_rho(sub, ref):
    agg = sub.groupby("name")["mean"].mean()
    common = [c for c in ref if c in agg.index]
    rho, p = st.spearmanr([ref[c] for c in common], [agg[c] for c in common])
    return rho, p, len(common)


def _report(label, sub, ref, score, full_steps):
    """full_steps: the longest rollout seen anywhere for this run/body, used as
    the "what a non-falling episode looks like" reference -- checkpoints that
    fall well short of it mean the policy was falling early in the rollout,
    so the correlation is computed from a handful of flailing steps, not
    stable gait, and should not be trusted the same way. (Found the hard way:
    an h2190 checkpoint reported as the single strongest correlation in the
    project turned out to be ~39 real steps per episode before a fall.)"""
    rho, p, n = _channel_rho(sub, ref)
    steps = sub["n_steps"].mean()
    frac = steps / full_steps if full_steps else 1.0
    flag = "  ** SHORT ROLLOUT (policy fell early) -- rho unreliable **" if frac < 0.8 else ""
    print(f"{label:>14s}  rho={rho:+.3f}  p={p:.4f}  n={n}  episode_score={score:.1f}  steps={steps:.0f}/{full_steps:.0f}{flag}")
    return rho


def channel_vs_real(csv_path, checkpoint_sweep):
    df = pd.read_csv(csv_path)
    ref = real_channel_reference()
    decoded = df[df.kind == "decoded_emg"]
    # body prefix (e.g. "h1622" out of "h1622_k11_w01_..._seed0") -- CSVs that
    # pool multiple bodies (generalization runs) must be reported per body,
    # never pooled together, or the correlation is meaningless.
    body = decoded["run"].str.extract(r"^(h\d+)")[0]

    if checkpoint_sweep:
        for b, body_sub in decoded.groupby(body):
            full_steps = body_sub["n_steps"].max()
            for step, sub in body_sub.groupby("checkpoint_step"):
                score = df[(df.kind == "episode_score") & (df.checkpoint_step == step) & (df["run"].str.startswith(b))]["mean"].mean()
                _report(f"{b} step {step}", sub, ref, score, full_steps)
    else:
        for b, sub in decoded.groupby(body):
            score = df[(df.kind == "episode_score") & (df["run"].str.startswith(b))]["mean"].mean()
            _report(b, sub, ref, score, sub["n_steps"].max())


# canonical k=11, w=0.1, hard-constraint, full-reward, clip, real config per
# body -- same setup the h0918 magnitude-inversion finding itself used, so the
# coverage numbers below line up with the CSV's own conditions, not a
# different config's actuator set.
DEFAULT_BODY_CONFIGS = {
    "h0918": "baselines_DEPRL/emg_lap/h0918/h0918_k11_w01_mirror_clip_net256_full_seed0/config.yaml",
    "h1622": "baselines_DEPRL/emg_lap/h1622/h1622_k11_w01_mirror_clip_net256_full_seed0/config.yaml",
    "h2190": "baselines_DEPRL/emg_lap/h2190/h2190_k11_w01_mirror_clip_net256_full_seed0/config.yaml",
}


def actuator_coverage(config_path):
    """(n_actuators, n_mapped, coverage_ratio) for one body, read directly off
    a freshly-built LatentActionPriorWrapper -- no agent/checkpoint needed."""
    import yaml
    from leaps.scripts.backfill_rl_checkpoint_metrics import build_env

    with open(config_path) as f:
        config = yaml.safe_load(f)
    _, inner_environment = build_env(config)
    n_actuators = inner_environment.n_actuators
    n_mapped = len(inner_environment._mapped_names)
    return n_actuators, n_mapped, n_mapped / n_actuators


def coverage_ratio(csv_path, body_configs):
    df = pd.read_csv(csv_path)
    ref = real_channel_reference()
    decoded = df[df.kind == "decoded_emg"]
    body = decoded["run"].str.extract(r"^(h\d+)")[0]

    rows = []
    for b, cfg in body_configs.items():
        sub = decoded[body == b]
        if sub.empty:
            print(f"{b}: no decoded_emg rows in {csv_path}, skipping")
            continue
        rho, p, n = _channel_rho(sub, ref)
        n_act, n_mapped, cov = actuator_coverage(cfg)
        rows.append((b, n_act, n_mapped, cov, rho, p))

    print(f"{'body':>8s}{'n_actuators':>13s}{'n_mapped':>10s}{'coverage':>10s}{'rho':>9s}{'|rho|':>8s}{'p':>8s}")
    for b, n_act, n_mapped, cov, rho, p in rows:
        print(f"{b:>8s}{n_act:>13d}{n_mapped:>10d}{cov:>10.3f}{rho:>+9.3f}{abs(rho):>8.3f}{p:>8.4f}")

    if len(rows) >= 3:
        cov_vals = [r[3] for r in rows]
        abs_rho_vals = [abs(r[4]) for r in rows]
        rho_corr, p_corr = st.spearmanr(cov_vals, abs_rho_vals)
        print(f"\ncoverage vs |rho| across bodies: spearman rho={rho_corr:+.3f}, p={p_corr:.4f}, n={len(rows)}"
              f"  (n=3 -- indicative at best, not a real significance test)")


def real_vs_null(csv_path, kind):
    df = pd.read_csv(csv_path)
    sub = df[df.kind == kind]
    real = sub[sub.run.str.startswith("real")].groupby("name")["mean"].mean()
    null = sub[sub.run.str.startswith("null")].groupby("name")["mean"].mean()
    if real.empty or null.empty:
        raise SystemExit(
            f"no real/null rows found for kind={kind!r} -- CSV must be built with "
            f"--run-label real_seedN / null_seedN (default run labels collide "
            f"between emg_lap and no_lap configs of the same body/k/w/seed)"
        )
    common = sorted(set(real.index) & set(null.index), key=lambda n: -real[n])
    print(f"{'name':22s}{'real':>12s}{'null':>12s}")
    for n in common:
        print(f"{n:22s}{real[n]:12.4f}{null[n]:12.4f}")
    rho, p = st.spearmanr([real[n] for n in common], [null[n] for n in common])
    print(f"\nSpearman rho(real,null)={rho:+.3f}  p={p:.4f}  n={len(common)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    p1 = sub.add_parser("channel-vs-real")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--checkpoint-sweep", action="store_true")

    p2 = sub.add_parser("real-vs-null")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--kind", required=True, help="e.g. residual_unmapped, prior_abs_mapped, decoded_emg")

    p3 = sub.add_parser("coverage-ratio")
    p3.add_argument("--csv", required=True)
    p3.add_argument("--config", action="append", default=[], metavar="BODY=PATH",
                     help="override a body's config path, e.g. --config h2190=path/to/config.yaml "
                          "(default: canonical k11/w01/hardconstraint/full/clip config per body)")

    args = p.parse_args()
    if args.mode == "channel-vs-real":
        channel_vs_real(args.csv, args.checkpoint_sweep)
    elif args.mode == "real-vs-null":
        real_vs_null(args.csv, args.kind)
    else:
        body_configs = dict(DEFAULT_BODY_CONFIGS)
        for override in args.config:
            body, path = override.split("=", 1)
            body_configs[body] = path
        coverage_ratio(args.csv, body_configs)


if __name__ == "__main__":
    main()
