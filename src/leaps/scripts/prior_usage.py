"""How the trained policy uses the frozen EMG decoder at t* (Results diagnostic).

  1. occupancy : where the latent action goes -- tanh saturation of u_l, and
                 whether the decoded pattern stays on the AB06 EMG manifold
                 (Mahalanobis of a_l vs the training latents; nearest-neighbour
                 distance of D(a_l) to AB06 training frames vs held-out frames)
  2. timing    : gait-cycle r of the prior a_hat, the blended excitation and
                 the activation vs the 22-subject EMG (same r as F2)
  3. share     : on covered muscles, (1-w)a_hat vs w|a_full|

Untrained prior is left out: its decoder is re-drawn unseeded on every env
build, so no rollout sees the decoders it was trained with.

    python -m leaps.scripts.prior_usage rollout --body h0918 --w w01 --seed seed0
    python -m leaps.scripts.prior_usage analyze
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.spatial import cKDTree

from leaps.data import time_normalize_stride
from leaps.envs.emg_mapping import EMG_CHANNELS, MODEL_MAPS

ROOT = Path(__file__).resolve().parents[3] / "results"
SEL = ROOT / "checkpoint_selection"
OUT = ROOT / "prior_usage"
DEC = ROOT / "synergy_priors/single_subject/decoder_k6_AB06_corrected"
BODIES, WS = ("h0918", "h1622", "h2190"), ("w00", "w01", "w05")
W_VAL = {"w00": 0.0, "w01": 0.1, "w05": 0.5}
N_EP, DT = 8, 0.025
MIN_STEPS = {"h2190": int(6 / DT)}   # h2190: episodes >= 6 s, else no falls
CHI2_99 = stats.chi2.ppf(0.99, 6)


def _selection(body, w):
    return list(csv.DictReader(open(SEL / f"emg_lap_{body}_{w}_onlyvelrew.csv")))


def _latent_wrapper(env):
    from leaps.envs.latent_env import LatentActionPriorWrapper
    e = env
    while not isinstance(e, LatentActionPriorWrapper):
        e = e.env
    return e


# ------------------------------------------------------------------ rollout
def rollout(body, w, seed):
    from leaps.scripts.reeval_checkpoint import build_env_and_agent, find_ts_dir

    row = next(r for r in _selection(body, w) if r["seed"] == seed)
    step = int(row["t_star"])
    env, agent = build_env_and_agent(find_ts_dir(row["run_dir"], step), step)
    W = _latent_wrapper(env)
    k = W.dim_latent
    model = env.unwrapped.model
    right_leg = next(l for l in model.legs() if l.name().endswith("_r"))
    cap = getattr(env, "max_episode_steps", 1000)

    rec = {x: [] for x in ("ep", "u_l", "a_l", "emg", "a_hat", "a_full", "a", "act", "grf_r")}
    for i in range(N_EP):
        env.seed(1000 + i)
        obs, n = env.reset(), 0
        ms, done = env.muscle_states, False
        while not done and n < cap:
            u = agent.test_step(obs, muscle_states=ms, steps=1e6)
            u = u[0] if getattr(u, "ndim", 1) > 1 else u
            obs, _, done, _ = env.step(u)
            ms = env.muscle_states
            a_l = W._to_z_domain(np.asarray(u[:k], np.float32))
            with torch.no_grad():
                emg = W.decoder(torch.from_numpy(a_l)[None])[0].numpy()
            for key, v in (("ep", i), ("u_l", u[:k]), ("a_l", a_l), ("emg", emg),
                           ("a_hat", W._last_a_hat), ("a_full", u[k:]),
                           ("a", np.clip(W._last_final_action, 0, 1)),
                           ("act", model.muscle_activation_array()),
                           ("grf_r", right_leg.contact_load())):
                rec[key].append(np.array(v, np.float32))
            n += 1
        print(f"{body} {w} {seed} ep{i}: {n} steps", flush=True)

    names = [a.name() for a in model.actuators()]
    OUT.joinpath("raw").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "raw" / f"{body}_{w}_{seed}.npz",
                        names=np.array(names), mapped=W._mapped_mask, t_star=step,
                        **{x: np.stack(v) for x, v in rec.items()})


# ------------------------------------------------------------------ analysis
def _ab06_frames():
    """AB06 unit-scaled frames, same split/normalisation as the decoder."""
    from leaps.data.metadata import ALL_MODES, LEAPS_H5_PATH
    from leaps.data.stride_dataset import load_strides
    strides = load_strides(LEAPS_H5_PATH, subjects=["AB06"], modes=ALL_MODES)
    idx = np.random.default_rng(0).permutation(len(strides))
    n_val = max(1, int(len(strides) * 0.2))
    p99 = np.load(DEC / "norm.npz")["p99"]
    unit = lambda s: np.clip(s / p99, 0, 1).reshape(-1, 11)
    return unit(strides[idx[n_val:]]), unit(strides[idx[:n_val]])


def _cycles(grf, ep, min_steps):
    """(start, end) index pairs of right strides, per kept episode."""
    out = []
    for e in np.unique(ep):
        sl = np.where(ep == e)[0]
        if len(sl) < min_steps:
            continue
        g = grf[sl]
        on = g > 0.05 * g.max()
        rise = np.where(on[1:] & ~on[:-1])[0] + 1
        out += [(sl[a], sl[b]) for a, b in zip(rise[:-1], rise[1:])
                if 0.6 <= (b - a) * DT <= 1.8]
    return out


def _mean_cycle(sig, cyc):
    return np.stack([time_normalize_stride(sig[a:b, None])[:, 0] for a, b in cyc]).mean(0)


def analyze():
    from leaps.scripts.biomech_fidelity import cross_human

    train, held = _ab06_frames()
    tree = cKDTree(train)
    nn_ref = np.percentile(tree.query(held)[0], 95)   # held-out -> train, 95th pct
    norm = np.load(DEC / "norm.npz")
    L = norm["whiten_A"] / 3.0                          # Sigma = L L^T
    Sinv, mu = np.linalg.inv(L @ L.T), norm["whiten_mu"]
    chn, sub_means, ch_idx, pair_r, _ = cross_human(1.2, 0.15)   # F2's reference

    occ, tim = [], []
    for body in BODIES:
        for w in WS:
            wv = W_VAL[w]
            for f in sorted((OUT / "raw").glob(f"{body}_{w}_seed*.npz")):
                d = np.load(f)
                seed = f.stem.split("_")[-1]
                dz = d["a_l"] - mu
                m2 = np.einsum("ij,jk,ik->i", dz, Sinv, dz)
                nn = tree.query(d["emg"])[0]
                cov = d["mapped"]
                ah, af = d["a_hat"][:, cov], np.abs(d["a_full"][:, cov])
                prior = (1 - wv) * ah
                occ.append(dict(
                    body=body, w=wv, seed=seed, steps=len(d["ep"]),
                    sat_frac=float(np.mean(np.abs(d["u_l"]) > 0.99)),
                    mahal_gt_chi2_99=float(np.mean(m2 > CHI2_99)),
                    mahal_median=float(np.median(np.sqrt(m2))),
                    offmanifold_frac=float(np.mean(nn > nn_ref)),
                    prior_share=float(prior.sum() / (prior.sum() + wv * af.sum() + 1e-8)),
                    abs_dev_from_prior=float(np.mean(np.abs(d["a"][:, cov] - ah)))))

                cyc = _cycles(d["grf_r"], d["ep"], MIN_STEPS.get(body, 1000))
                if not cyc:
                    continue
                names = list(d["names"])
                for ch, musc in MODEL_MAPS[body].items():
                    j, i = names.index(musc), ch_idx[ch]
                    r = {}
                    for key in ("a_hat", "a", "act"):
                        mc = _mean_cycle(d[key][:, j], cyc)
                        r[key] = float(np.mean([stats.pearsonr(mc, sub_means[s][:, i])[0]
                                                for s in sub_means]))
                    tim.append(dict(body=body, w=wv, seed=seed, channel=ch, muscle=musc,
                                    n_cyc=len(cyc), r_prior=r["a_hat"], r_excitation=r["a"],
                                    r_activation=r["act"],
                                    human_p5=float(np.percentile(pair_r[ch], 5))))

    for name, rows in (("occupancy", occ), ("timing", tim)):
        with open(OUT / f"{name}.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    print(f"nn_ref (held-out 95th pct) = {nn_ref:.4f}; wrote {OUT}/occupancy.csv, timing.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["rollout", "analyze"])
    p.add_argument("--body")
    p.add_argument("--w")
    p.add_argument("--seed")
    a = p.parse_args()
    rollout(a.body, a.w, a.seed) if a.stage == "rollout" else analyze()
