"""
TRUE-RANK sweep: is low-rank the load-bearing assumption for CF's advantage?

Cycle 6 showed reward sharpening (which raises effective rank) collapses the CF
advantage. This tests it cleanly: vary the TRUE rank d of R = P U^T (P:m*d,
U:n*d, unit rows), with CF given the ORACLE rank (latent_dim = true d). Predict
CF/Tabular advantage decreases monotonically as d -> min(m,n) (R becomes full
rank), because CF must then learn d(m+n) ~ mn parameters, losing its
sample-efficiency edge, while tabular is unchanged.

This is the Set N (structural stress test) redo done in the CORRECT regime:
sample-starved with changing availability and reward-sharing (p=1), where the
low-rank assumption actually binds.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_refit import Tabular, RefitMF, run_episode
from pilot_structure import oracle_rand, _normrows


def make_world_rank(m, n, d, seed):
    rng = np.random.RandomState(seed)
    P = _normrows(rng.randn(m, d))
    U = _normrows(rng.randn(n, d))
    return P, U, P @ U.T


def skill(Cls, hp, m, n, d, T, cand, sigma, p, seeds):
    sk, gg = [], []
    for s in seeds:
        w = make_world_rank(m, n, d, s)
        _, g, _, _ = run_episode(Cls, hp, w, T, p, s, sigma, cand)
        oc, rd = oracle_rand(w[2], s, cand)
        sk.append((g - rd) / max(oc - rd, 1e-6)); gg.append(g)
    return np.mean(sk), np.mean(gg)


def main():
    m, n, T, cand, sigma, p = 30, 120, 50, 15, 0.10, 1.0
    seeds = list(range(3))
    tab_hp = dict(eps=0.15)
    cf_hp = dict(use_choices=False, eps=0.15, als_sweeps=5, refit_every=3)
    ranks = [1, 2, 3, 5, 8, 15, 30]

    print("=" * 80)
    print("TRUE-RANK SWEEP (p=1, starved). m=%d n=%d T=%d cand=%d %d seeds. "
          "CF uses oracle rank." % (m, n, T, cand, len(seeds)))
    print("=" * 80)
    print(f"{'true rank d':>11s}  {'Tab skill':>9s}  {'CF skill':>9s}  {'CF-Tab':>7s}  {'CF/Tab(greedy)':>14s}")
    print("-" * 80)
    for d in ranks:
        ts, tg = skill(Tabular, tab_hp, m, n, d, T, cand, sigma, p, seeds)
        cs, cg = skill(RefitMF, cf_hp, m, n, d, T, cand, sigma, p, seeds)
        print(f"{d:11d}  {ts:9.3f}  {cs:9.3f}  {cs-ts:7.3f}  {cg/tg if tg>1e-9 else float('nan'):14.2f}")
    print("-" * 80)
    print("PREDICT: CF-Tab gap shrinks as d rises (full rank => no low-rank edge).")


if __name__ == "__main__":
    main()
