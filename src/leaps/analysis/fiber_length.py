"""Operating range of the muscles: how far the fibers work from their optimal length. Same rollouts,
episode filter and checkpoint choice as gait_metrics.py, so it adds a column to GAIT_METRICS
without changing it.

  dev_opt   mean |l~ - 1| over time and muscles, l~ = fiber_length_norm (1 = optimal length)
  in_plateau  fraction of time-muscle samples with 0.8 <= l~ <= 1.2 (near the top of the force-length curve)

    python -m leaps.analysis.fiber_length
"""
import csv

import numpy as np

from leaps.scripts.biomech_fidelity import MIN_ROWS, RUNS, load_tstar_map, read_sto
from leaps.scripts.gait_metrics import METHODS, OUT, _ckpt_dir


def main():
    tmap = load_tstar_map()
    rows = []
    for body in ("h0918", "h1622", "h2190"):
        min_gait = MIN_ROWS.get(body, 2500) * 0.4
        for key, label in METHODS.items():
            dev, plat, n_ep = [], [], 0
            for rd in RUNS[body].get(key, []):
                ck = _ckpt_dir(rd, tmap)
                if ck is None:
                    continue
                for sto in sorted(ck.glob("[0-9]*.sto")):
                    ep = read_sto(sto)
                    if len(ep) < min_gait:
                        continue
                    cols = [c for c in ep.columns if c.endswith(".fiber_length_norm")]
                    n = len(ep)
                    L = ep[cols].to_numpy()[int(0.05 * n):int(0.95 * n)]
                    dev.append(np.mean(np.abs(L - 1)))
                    plat.append(np.mean((L >= 0.8) & (L <= 1.2)))
                    n_ep += 1
            if n_ep:
                rows.append(dict(body=body, method=label, n_ep=n_ep, dev_opt=round(float(np.mean(dev)), 3),
                                 in_plateau=round(float(np.mean(plat)), 3)))
                print(rows[-1])
    with open(OUT / "FIBER_LENGTH.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / 'FIBER_LENGTH.csv'}")


if __name__ == "__main__":
    main()
