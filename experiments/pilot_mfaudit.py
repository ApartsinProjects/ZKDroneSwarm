"""
MF APPROXIMATION AUDIT: are our results limited by factorization approximations
(under-convergence, regularization) rather than by data/fundamentals?

Tabular uses EXACT empirical means; only CF has factorization approximations, so
any CF under-fit biases every CF-vs-tabular comparison. We check:
  1. ALS convergence: vary als_sweeps x refit_every. If CF skill is FLAT across
     more compute, it is converged (results sound). If it keeps rising, current
     settings UNDERSTATE CF.
  2. Regularization: vary prior_prec. If skill is insensitive, not over/under-
     regularized.
Clean regime (sigma=0.1, p=1) to isolate the factorization from noise/exploration.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_noise import Tabular, RewardCF, skill_ms

cfg = dict(structure="cluster_gauss", eps=0.10, n_clusters=15, sharp=1.0)


def main():
    m, d, T, cand = 30, 5, 50, 20
    so = sb = 0.10
    seeds = list(range(3))
    base = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)

    print("=" * 90)
    print("MF AUDIT (p=1, clean sigma=0.10, clustG nc15). m=%d d=%d T=%d cand=%d %d seeds" % (m, d, T, cand, len(seeds)))
    print("=" * 90)

    for n in [120, 240]:
        print(f"\n--- n={n} ---")
        tabm, _ = skill_ms(Tabular, dict(**base), cfg, m, n, d, T, cand, so, sb, seeds)
        print(f"  Tabular (exact, reference): {tabm:.3f}")
        print(f"  {'als_sweeps':>10s} {'refit_every':>11s} {'prior_prec':>10s} | {'CF skill':>9s}")
        print("  " + "-" * 50)
        # convergence sweep (prior_prec fixed=1.0)
        for sweeps, refit in [(2, 3), (5, 3), (10, 2), (20, 1), (40, 1)]:
            hp = dict(als_sweeps=sweeps, refit_every=refit, prior_prec=1.0, **base)
            cm, _ = skill_ms(RewardCF, hp, cfg, m, n, d, T, cand, so, sb, seeds)
            print(f"  {sweeps:10d} {refit:11d} {1.0:10.1f} | {cm:9.3f}")
        # regularization sweep (sweeps fixed=20, refit_every=1)
        for pp in [0.1, 0.3, 1.0, 3.0]:
            hp = dict(als_sweeps=20, refit_every=1, prior_prec=pp, **base)
            cm, _ = skill_ms(RewardCF, hp, cfg, m, n, d, T, cand, so, sb, seeds)
            print(f"  {20:10d} {1:11d} {pp:10.1f} | {cm:9.3f}")
    print("\n" + "=" * 90)
    print("READ: if CF skill is FLAT across sweeps/refit (esp. 5,3 vs 40,1), our default")
    print("(als_sweeps=5, refit_every=3) is converged -> results are NOT approximation-limited.")
    print("If 40,1 >> 5,3, we are understating CF and must rerun key experiments at higher compute.")


if __name__ == "__main__":
    main()
