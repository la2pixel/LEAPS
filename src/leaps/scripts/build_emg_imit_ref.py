"""22-subject RVC cycle-mean activation reference for the muscle-space imitation
reward (gaitgym._emg_imitation_reward). Same subject/speed filtering as
biomech_fidelity.cross_human. One (101,) array per EMG channel."""
import h5py
import numpy as np

H5 = "/home/nadinebadie/lalitha/datasets/emg_activations_v2.h5"
OUT = "/home/nadinebadie/lalitha/LEAPS/results/emg_driven/emg_imit_ref_22subj.npz"
SPEED, TOL = 1.2, 0.15

with h5py.File(H5, "r") as f:
    chans = [c.decode() if isinstance(c, bytes) else str(c) for c in f.attrs["emg_channels"]]
    subs = [s for s in f.keys() if s.startswith("AB")]
    per_sub = []
    for s in subs:
        g = f[s]
        modes = g["modes"][:].astype(str)
        cond = g["conditions"][:].astype(str)
        spd = np.array([float(x) if x.replace(".", "", 1).isdigit() else np.nan for x in cond])
        keep = (modes == "treadmill") & (np.abs(spd - SPEED) <= TOL)
        if keep.sum() < 5:
            continue
        m = g["strides"][keep].mean(axis=0)                          # (101, 11)
        peak = np.abs(m).max(axis=0)
        m = m / np.where(peak < 1e-9, 1.0, peak)                     # per-channel RVC
        per_sub.append(m)

ref = np.stack(per_sub).mean(axis=0)                                 # (101, 11)
np.savez(OUT, **{c: ref[:, i].astype(np.float64) for i, c in enumerate(chans)})
print(f"wrote {OUT}  ({len(per_sub)} subjects, channels: {chans})")
