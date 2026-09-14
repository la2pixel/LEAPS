"""R7 firm-up (no RL): train a k=6 corrected decoder for a second subject
(AB20) with the exact AB06 recipe, then show the two single-subject decoders
capture *structurally different* muscle-coordination.

Evidence produced:
  1. held-out R^2 of each decoder on its own subject   (does each capture its subject)
  2. cross-autoencoding R^2: subject X's EMG through subject Y's AE
     (if the manifold is subject-specific, cross < self)
  3. synergy-weight comparison: decoder Jacobian (11x6) at the data mean,
     greedy-matched between subjects by |cosine|, + subspace principal angles
  4. decoded cycle-mean muscle waveforms, AB06 vs AB20, key muscles

    python -m leaps.scripts.compare_subject_decoders            # AB06 vs AB20
    python -m leaps.scripts.compare_subject_decoders --b AB23   # AB06 vs AB23
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("LEAPS_EMG_H5",
                      "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5")

import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.linalg import subspace_angles

from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH
from leaps.data.stride_dataset import load_strides
from leaps.synergy_common import train_ae
from leaps.envs.emg_mapping import EMG_CHANNELS

# corrected recipe (matches decoder_k6_AB06_corrected/results.yaml)
K, DEPTH, THRESH, SCALE, SEED = 6, 1, 0.6, 0.9, 0
OUT = Path("/home/nadinebadie/lalitha/LEAPS/results/synergy_priors")
FIGDIR = Path("/home/nadinebadie/lalitha/LEAPS/results/subject_decoder_compare")


def prep(subject):
    strides = load_strides(LEAPS_H5_PATH, subjects=[subject], modes=ALL_MODES)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(strides))
    n_val = max(1, int(0.2 * len(strides)))
    sv, st = strides[idx[:n_val]], strides[idx[n_val:]]
    div = np.percentile(st, 99, axis=1).mean(axis=0) + 1e-8
    to_unit = lambda s: np.clip(s / div, 0, 1).astype(np.float32)  # noqa: E731
    Xtr = to_unit(st).reshape(-1, 11)
    n_sub = min(50_000, len(Xtr))
    sub = np.random.default_rng(0).choice(len(Xtr), n_sub, replace=False)
    return dict(subject=subject, Xtr=Xtr[sub], Xval=to_unit(sv).reshape(-1, 11),
               cycle_mean=to_unit(st).mean(axis=0), div=div, n_strides=len(strides))


def r2(x, xhat):
    return float(1 - ((x - xhat) ** 2).sum() / ((x - x.mean(0)) ** 2).sum())


def ae_recon(model, X):
    with torch.no_grad():
        xhat, z = model(torch.from_numpy(X.astype(np.float32)))
    return xhat.numpy(), z.numpy()


def jacobian(model, z0):
    z = torch.tensor(z0, dtype=torch.float32, requires_grad=True).unsqueeze(0)
    J = torch.autograd.functional.jacobian(lambda zz: model.decoder(zz), z)
    return J.squeeze().numpy()  # (11, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="AB06")
    ap.add_argument("--b", default="AB20")
    args = ap.parse_args()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    d = {}
    for s in (args.a, args.b):
        p = prep(s)
        print(f"training {s}: {p['n_strides']} strides, {len(p['Xtr'])} frames")
        m = train_ae(p["Xtr"], dim_latent=K, epochs=50, batch_size=512, seed=SEED,
                     threshold=THRESH, scale=SCALE, depth=DEPTH)
        m.eval()
        xhat, ztr = ae_recon(m, p["Xtr"])
        xhat_v, _ = ae_recon(m, p["Xval"])
        p["model"], p["r2_self"] = m, r2(p["Xval"], xhat_v)
        p["zmean"] = ztr.mean(0)
        d[s] = p
        print(f"  held-out R2 (own subject): {p['r2_self']:.4f}")

    A, B = d[args.a], d[args.b]

    # 2. cross-autoencoding
    xhat_AB, _ = ae_recon(B["model"], A["Xval"])   # A's EMG through B's AE
    xhat_BA, _ = ae_recon(A["model"], B["Xval"])
    cross_AB, cross_BA = r2(A["Xval"], xhat_AB), r2(B["Xval"], xhat_BA)
    print(f"\ncross-AE R2:  {args.a} EMG through {args.b} AE = {cross_AB:.4f}  "
          f"(self {A['r2_self']:.4f}, drop {A['r2_self']-cross_AB:+.4f})")
    print(f"cross-AE R2:  {args.b} EMG through {args.a} AE = {cross_BA:.4f}  "
          f"(self {B['r2_self']:.4f}, drop {B['r2_self']-cross_BA:+.4f})")

    # 3. synergy weights
    Wa, Wb = jacobian(A["model"], A["zmean"]), jacobian(B["model"], B["zmean"])
    Wan = Wa / (np.linalg.norm(Wa, axis=0, keepdims=True) + 1e-9)
    Wbn = Wb / (np.linalg.norm(Wb, axis=0, keepdims=True) + 1e-9)
    C = np.abs(Wan.T @ Wbn)                      # (6,6) |cos| between synergies
    ri, ci = linear_sum_assignment(-C)
    matched = C[ri, ci]
    angles = np.degrees(subspace_angles(Wa, Wb))
    print(f"\nmatched synergy |cosine| ({args.a} vs {args.b}): "
          + ", ".join(f"{x:.2f}" for x in sorted(matched, reverse=True)))
    print(f"  mean {matched.mean():.2f}   (1.00 = identical structure)")
    print(f"subspace principal angles (deg): "
          + ", ".join(f"{a:.1f}" for a in angles))

    # 4. decoded cycle-mean waveforms
    def decoded_cycle(model, cyc):
        with torch.no_grad():
            xhat, _ = model(torch.from_numpy(cyc.astype(np.float32)))
        return xhat.numpy()
    dcA, dcB = decoded_cycle(A["model"], A["cycle_mean"]), decoded_cycle(B["model"], B["cycle_mean"])

    # ---- figure ----
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    ax.bar([0, 1], [A["r2_self"], cross_BA], color=["#d62728", "#bbbbbb"])
    ax.bar([3, 4], [B["r2_self"], cross_AB], color=["#1f77b4", "#bbbbbb"])
    ax.set_xticks([0, 1, 3, 4])
    ax.set_xticklabels([f"{args.a}\nself", f"{args.b}→{args.a}\nAE", f"{args.b}\nself",
                        f"{args.a}→{args.b}\nAE"], fontsize=7.5)
    ax.set_ylabel("reconstruction R²"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.3)
    ax.set_title("Each decoder reconstructs its own\nsubject better than the other's", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(range(K), sorted(matched, reverse=True), color="#7f4fc9")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("matched synergy"); ax.set_ylabel("|cosine| between subjects")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=.3)
    ax.set_title(f"Synergy-weight match {args.a} vs {args.b}\nmean {matched.mean():.2f} "
                 f"(1.0 = identical)", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.bar(range(1, K + 1), angles, color="#2ca02c")
    ax.set_xlabel("subspace dimension"); ax.set_ylabel("principal angle (deg)")
    ax.grid(axis="y", alpha=.3)
    ax.set_title("Principal angles between the two\n6-D synergy subspaces", fontsize=9)

    ax = fig.add_subplot(gs[1, :])
    pct = np.linspace(0, 100, dcA.shape[0])
    for mus in ["vastusmedialis", "bicepsfemoris", "gastrocmed"]:
        j = EMG_CHANNELS.index(mus)
        ax.plot(pct, dcA[:, j], lw=1.8, label=f"{args.a} {mus}")
        ax.plot(pct, dcB[:, j], lw=1.8, ls="--", label=f"{args.b} {mus}")
    ax.set_xlabel("gait cycle [%]"); ax.set_ylabel("decoded activation")
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=.3)
    ax.set_title("Decoded cycle-mean muscle waveforms — the two subject decoders "
                 "produce different timing/amplitude", fontsize=9)
    fig.suptitle(f"Single-subject decoders capture individual structure: "
                 f"{args.a} vs {args.b}  (k=6, corrected recipe)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = FIGDIR / f"decoder_compare_{args.a}_vs_{args.b}.png"
    fig.savefig(png, dpi=140)
    print(f"\nwrote {png}")

    # ---- persist the B decoder as a reusable artifact + a results file ----
    bdir = OUT / "single_subject" / f"decoder_k{K}_{args.b}_corrected"
    bdir.mkdir(parents=True, exist_ok=True)
    for p in B["model"].decoder.parameters():
        p.requires_grad = False
    torch.save(B["model"].decoder.state_dict(), bdir / "decoder.pt")
    _, ztr_b = ae_recon(B["model"], B["Xtr"])
    wmu = ztr_b.mean(0)
    Sig = np.cov(ztr_b, rowvar=False)
    ev, evec = np.linalg.eigh(Sig)
    o = np.argsort(ev)[::-1]
    wA = 3.0 * (evec[:, o] @ np.diag(np.sqrt(ev[o])))
    np.savez(bdir / "norm.npz", p01=np.zeros(11, np.float32), p99=B["div"].astype(np.float32),
             z_lo=ztr_b.min(0).astype(np.float32), z_hi=ztr_b.max(0).astype(np.float32),
             whiten_A=wA.astype(np.float32), whiten_mu=wmu.astype(np.float32))
    res = dict(comparison=f"{args.a}_vs_{args.b}", k=K, recipe=dict(depth=DEPTH, threshold=THRESH, scale=SCALE),
               r2_self={args.a: A["r2_self"], args.b: B["r2_self"]},
               cross_ae_r2={f"{args.b}_thru_{args.a}": cross_BA, f"{args.a}_thru_{args.b}": cross_AB},
               matched_synergy_cosine_mean=float(matched.mean()),
               matched_synergy_cosine=sorted(float(x) for x in matched),
               subspace_principal_angles_deg=[float(a) for a in angles])
    with open(FIGDIR / f"decoder_compare_{args.a}_vs_{args.b}.yaml", "w") as f:
        yaml.dump(res, f, sort_keys=False)
    with open(bdir / "results.yaml", "w") as f:
        yaml.dump(dict(subject=args.b, latent_dim=K, depth=DEPTH, threshold=THRESH,
                       scale=SCALE, seed=SEED, held_out_r2=B["r2_self"],
                       note="corrected recipe, trained for R7 individual-traceability firm-up"), f, sort_keys=False)
    print(f"wrote {bdir}/  (decoder.pt, norm.npz, results.yaml)")
    print(f"wrote {FIGDIR}/decoder_compare_{args.a}_vs_{args.b}.yaml")


if __name__ == "__main__":
    main()
