"""
C1: CF VARIANT BAKE-OFF across the core design space (structure x observability).

Re-anchored scope: swarm + unknown latent structure + limited/noisy observability
+ no communication; honest agents; method = collaborative filtering via matrix
decomposition. Question: which CF variant DOMINATES the design space, and is
there a single SIMPLE method that wins everywhere?

Variants:
  Tabular   - no decomposition (independent per-drone; baseline)
  RewardCF  - matrix decomposition from own + others' (noise-weighted) REWARDS
  ChoiceCF  - decomposition from own reward + competence-weighted CHOICES (decision)
  BothCF    - fuse both channels (use all available signals)

Design space:
  structure   = clustG nc15 (clustered low-rank) | gauss (continuous low-rank)
  observability = reward-clean (p=1, sigma_obs=0.1) | reward-noisy (p=1, 0.6)
                | decision-only (p=0; rewards private, choices public)
Compute set from MF audit: als_sweeps=10, refit_every=2.
Metric: skill = (greedy - random)/(oracle - random).
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_noise import (Tabular, RewardCF, ChoiceCF, BothCF, BothCFGated,
                         run_episode, make_world_struct, oracle_rand)


def skill(Cls, hp, cfg, p, so, sb, m, n, d, T, cand, seeds):
    sk = []
    for s in seeds:
        w = make_world_struct(m, n, d, cfg["structure"], cfg["eps"], cfg["n_clusters"], cfg["sharp"], s)
        g = run_episode(Cls, hp, w, T, p, s, so, sb, cand)
        oc, rd = oracle_rand(w[2], s, cand)
        sk.append((g - rd) / max(oc - rd, 1e-6))
    return float(np.mean(sk))


def main():
    m, n, d, T, cand = 30, 120, 5, 50, 20
    so = 0.10
    seeds = list(range(3))
    base = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=6, refit_every=3)
    methods = {
        "Tabular":  (Tabular,  dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)),
        "RewardCF": (RewardCF, dict(**base)),
        "ChoiceCF": (ChoiceCF, dict(comp=True, s2c=0.2, n_neg=1, within=True, warm_frac=0.3, T_total=T, **base)),
        "BothCF":   (BothCF,   dict(comp=True, s2c=0.2, n_neg=1, within=True, warm_frac=0.3, T_total=T, **base)),
        "BothGated": (BothCFGated, dict(comp=True, s2c=0.2, n_neg=1, within=True, warm_frac=0.3,
                                        T_total=T, gate_alpha=0.5, **base)),
    }
    structures = [
        ("clustG nc15", dict(structure="cluster_gauss", eps=0.10, n_clusters=15, sharp=1.0)),
        ("gauss",       dict(structure="gauss",         eps=0.00, n_clusters=0,  sharp=1.0)),
    ]
    observ = [("reward-clean", 1.0, 0.10), ("reward-noisy", 1.0, 0.60), ("decision-only", 0.0, 0.10)]

    print("=" * 92)
    print("C1: CF VARIANT BAKE-OFF (structure x observability). m=%d n=%d d=%d T=%d cand=%d %d seeds"
          % (m, n, d, T, cand, len(seeds)))
    print("skill=(greedy-rand)/(oracle-rand). [*]=best in row. als_sweeps=%d,refit_every=%d."
          % (base["als_sweeps"], base["refit_every"]))
    print("=" * 92)
    names = list(methods.keys())
    hdr = f"{'structure':12s} {'observability':14s} | " + " ".join(f"{nm:>9s}" for nm in names)
    print(hdr); print("-" * len(hdr))
    for slabel, cfg in structures:
        for olabel, p, sb in observ:
            vals = {nm: skill(Cls, hp, cfg, p, so, sb, m, n, d, T, cand, seeds)
                    for nm, (Cls, hp) in methods.items()}
            best = max(vals, key=vals.get)
            cells = " ".join(f"{vals[nm]:8.3f}{'*' if nm == best else ' '}" for nm in names)
            print(f"{slabel:12s} {olabel:14s} | {cells}")
    print("-" * len(hdr))
    print("READ: a single variant winning [*] across ALL cells = the simple dominant method.")
    print("Expect RewardCF to win reward-clean; ChoiceCF robust under noise/decision-only;")
    print("BothCF (fuse all signals) the candidate to dominate everywhere.")


if __name__ == "__main__":
    main()
