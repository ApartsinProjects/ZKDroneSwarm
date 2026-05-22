"""
Multi-seed confirmation (error bars) of the two headline results:
  (a) p=1 (reward-shared): CF (RewardMF) >> Tabular in the sample-starved corner.
  (b) p=0 (decision-only): competence-weighted DualConf > Tabular (choices
      transfer knowledge with NO reward sharing).
Across two latent structures (onehot, cluster_gauss) for robustness, 5 seeds.
Metric: skill = (greedy - random)/(oracle - random), reported mean +/- std.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_refit import Tabular, RefitMF, run_episode
from pilot_structure import make_world_struct, oracle_rand
from pilot_bootstrap import DualConf


def skill_ms(Cls, hp, cfg, m, n, d, T, cand, sigma, p, seeds):
    sk = []
    for s in seeds:
        w = make_world_struct(m, n, d, cfg["structure"], cfg["eps"],
                              cfg["n_clusters"], cfg["sharp"], s)
        _, g, _, _ = run_episode(Cls, hp, w, T, p, s, sigma, cand)
        oc, rd = oracle_rand(w[2], s, cand)
        sk.append((g - rd) / max(oc - rd, 1e-6))
    return float(np.mean(sk)), float(np.std(sk))


def main():
    m, d, T, cand, sigma = 30, 5, 50, 20, 0.10
    seeds = list(range(5))
    tab = dict(eps=0.15)
    rmf = dict(use_choices=False, eps=0.15, als_sweeps=5, refit_every=3)
    conf = dict(als_sweeps=5, s2c=0.2, n_neg=1, refit_every=3, within=True,
                eps0=0.5, eps_min=0.05, eps_decay=0.93, warm_frac=0.3,
                T_total=T, use_consistency=True)
    structures = [
        ("onehot nc5",   dict(structure="onehot",        eps=0.20, n_clusters=5,  sharp=1.0)),
        ("clustG nc15",  dict(structure="cluster_gauss", eps=0.10, n_clusters=15, sharp=1.0)),
    ]
    print("=" * 100)
    print("CONFIRMATION (5 seeds). skill mean +/- std. m=%d d=%d T=%d cand=%d" % (m, d, T, cand))
    print("=" * 100)
    for label, cfg in structures:
        print(f"\n### structure = {label}")
        print(f"{'n':>5s} {'cov%':>5s} | {'Tab p1':>12s} {'CF p1':>12s} {'CF/Tab':>7s} | "
              f"{'Tab p0':>12s} {'DualConf p0':>13s} {'win%':>6s}")
        print("-" * 96)
        for n in [120, 240]:
            cov = 100.0 * min(1.0, T / n)
            t1m, t1s = skill_ms(Tabular, tab, cfg, m, n, d, T, cand, sigma, 1.0, seeds)
            c1m, c1s = skill_ms(RefitMF, rmf, cfg, m, n, d, T, cand, sigma, 1.0, seeds)
            t0m, t0s = skill_ms(Tabular, tab, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
            d0m, d0s = skill_ms(DualConf, conf, cfg, m, n, d, T, cand, sigma, 0.0, seeds)
            ratio = c1m / t1m if t1m > 1e-9 else float('nan')
            winp = 100.0 * (d0m / t0m - 1.0) if t0m > 1e-9 else float('nan')
            print(f"{n:5d} {cov:4.0f}% | {t1m:6.3f}+-{t1s:4.3f} {c1m:6.3f}+-{c1s:4.3f} {ratio:7.2f} | "
                  f"{t0m:6.3f}+-{t0s:4.3f} {d0m:6.3f}+-{d0s:4.3f} {winp:+5.0f}%")
    print("\n" + "=" * 100)
    print("(a) CF/Tab > 1 at p=1 = reward-observable win. (b) DualConf win% > 0 at p=0 =")
    print("decision-only win (choices transfer knowledge without reward sharing).")


if __name__ == "__main__":
    main()
