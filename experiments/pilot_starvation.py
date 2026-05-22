"""
Consolidation experiment: the CF-vs-tabular advantage as a 2D phase diagram over
(sample-starvation) x (reward-sharing).

Established across cycles 1-4:
  - Reward-observable + sample-starved: CF (reward-broadcast pooling + low-rank)
    beats independent tabular by 40-50%.
  - Sample-rich (any sharing): tie (explains prior Sets M/N negatives).
  - Decision-only (choices, no reward sharing): choices cannot beat tabular
    online (bootstrap deadlock; choice-recovery ceiling ~= tabular). Falsified.

This script produces the centerpiece: vary n (starvation, since coverage ~ T/n)
at fixed m, T; at reward-sharing p in {0, 1}; report CF (RewardMF) vs Tabular
final-greedy and the advantage ratio. Expect the CF advantage to be ~1 when
rich, rise as starvation increases at p=1, and stay ~1 at p=0.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_refit import Tabular, RefitMF, avg_seeds


def main():
    m, d, T, cand = 30, 5, 50, 15
    seeds = list(range(5))
    ns = [30, 60, 120, 240, 480]
    print("=" * 96)
    print("PHASE DIAGRAM: CF (RewardMF, reward-broadcast) vs Tabular, over "
          "starvation (n) x reward-sharing (p)")
    print(f"m={m}, d={d}, T={T}, cand={cand}, {len(seeds)} seeds. "
          "metric = final-model greedy reward (fraction of oracle).")
    print("=" * 96)
    print(f"{'n':>5s} {'cover%':>7s} | "
          f"{'p=1 Tab':>8s} {'p=1 CF':>8s} {'p=1 CF/Tab':>11s} | "
          f"{'p=0 Tab':>8s} {'p=0 CF':>8s} {'p=0 CF/Tab':>11s}")
    print("-" * 96)
    for n in ns:
        wp = dict(m=m, n=n, d=d, mode_eps=0.25, sigma=0.10, cand=cand)
        row = {}
        for p in (1.0, 0.0):
            _, _, tg, tf, _ = avg_seeds(Tabular, dict(eps=0.15), T, p, seeds, wp)
            _, _, cg, cf, _ = avg_seeds(
                RefitMF, dict(use_choices=False, eps=0.15, als_sweeps=5, refit_every=3),
                T, p, seeds, wp)
            row[p] = (tf, cf, cg / tg if tg > 1e-9 else float('nan'))
        cover = 100.0 * min(1.0, T / n)
        t1, c1, r1 = row[1.0]; t0, c0, r0 = row[0.0]
        print(f"{n:5d} {cover:6.0f}% | {t1:8.2f} {c1:8.2f} {r1:11.2f} | "
              f"{t0:8.2f} {c0:8.2f} {r0:11.2f}")
    print("-" * 96)
    print("Tab/CF columns are fraction-of-oracle; CF/Tab is the greedy-reward ratio.")
    print("READ: CF/Tab >> 1 only in the starved + reward-shared (p=1) corner;")
    print("~1 when rich (small n) or when rewards are private (p=0).")


if __name__ == "__main__":
    main()
