"""
Does LATENT STRUCTURE govern the CF-vs-tabular advantage?

All prior pilots fixed one structure (one-hot modes, eps=0.25 -> near-binary
rewards). This sweeps the latent geometry and measures how much CF beats tabular
in the reward-observable, sample-starved regime (p=1).

Knobs:
  - structure: 'onehot' (axis clusters), 'cluster_gauss' (random cluster
    centers), 'gauss' (continuous, no clusters).
  - eps: cluster tightness / latent noise (small=tight, large=diffuse).
  - n_clusters: number of clusters (vs d).
  - sharp: reward nonlinearity exponent (1=linear dot product, >1 = peaked, so
    few strongly-aligned targets per drone).

Metric: structure-INVARIANT skill = (greedy - random)/(oracle - random) in
[0,1], so reward scale/offset don't confound the comparison. Also reports oracle
and random levels (dynamic range = how 'easy' the structure is).

Hypothesis under test: CF's advantage over tabular is large for tight clusters
and/or peaked rewards (rare-but-predictable good targets), small for diffuse or
flat structures.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_refit import Tabular, RefitMF, run_episode


def _normrows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def make_world_struct(m, n, d, structure, eps, n_clusters, sharp, seed):
    rng = np.random.RandomState(seed)
    if structure == "gauss":
        P = _normrows(rng.randn(m, d)); U = _normrows(rng.randn(n, d))
    else:
        if structure == "onehot":
            nc = min(n_clusters, d)
            centers = np.eye(d)[:nc]
        else:  # cluster_gauss
            nc = n_clusters
            centers = _normrows(rng.randn(nc, d))
        dm = rng.randint(0, len(centers), m)
        tm = np.concatenate([np.arange(len(centers)),
                             rng.randint(0, len(centers), n - len(centers))])
        rng.shuffle(tm)
        P = _normrows(centers[dm] + eps * rng.randn(m, d))
        U = _normrows(centers[tm] + eps * rng.randn(n, d))
    R = P @ U.T
    if sharp != 1.0:
        R = np.sign(R) * np.abs(R) ** sharp
    return P, U, R


def oracle_rand(R, seed, cand, rounds=100):
    m, n = R.shape
    rng = np.random.RandomState(seed + 555)
    oc, rd = [], []
    for _ in range(rounds):
        for k in range(m):
            off = rng.choice(n, size=cand, replace=False)
            oc.append(R[k, off].max()); rd.append(R[k, off].mean())
    return float(np.mean(oc)), float(np.mean(rd))


def skill_over_seeds(Cls, hp, cfg, m, n, d, T, cand, sigma, p, seeds):
    sks, ocs, rds, gs = [], [], [], []
    for s in seeds:
        world = make_world_struct(m, n, d, cfg["structure"], cfg["eps"],
                                  cfg["n_clusters"], cfg["sharp"], s)
        _, g, _, _ = run_episode(Cls, hp, world, T, p, s, sigma, cand)
        oc, rd = oracle_rand(world[2], s, cand)
        denom = max(oc - rd, 1e-6)
        sks.append((g - rd) / denom); ocs.append(oc); rds.append(rd); gs.append(g)
    return np.mean(sks), np.mean(gs), np.mean(ocs), np.mean(rds)


def main():
    m, n, d, T, cand, sigma, p = 30, 120, 5, 50, 15, 0.10, 1.0
    seeds = list(range(3))
    tab_hp = dict(eps=0.15)
    cf_hp = dict(use_choices=False, eps=0.15, als_sweeps=5, refit_every=3)

    configs = [
        ("onehot eps.15 nc5",        dict(structure="onehot",        eps=0.15, n_clusters=5,  sharp=1.0)),
        ("onehot eps.35 nc5",        dict(structure="onehot",        eps=0.35, n_clusters=5,  sharp=1.0)),
        ("onehot eps.15 nc5 sharp3", dict(structure="onehot",        eps=0.15, n_clusters=5,  sharp=3.0)),
        ("clustG eps.10 nc5",        dict(structure="cluster_gauss", eps=0.10, n_clusters=5,  sharp=1.0)),
        ("clustG eps.35 nc5",        dict(structure="cluster_gauss", eps=0.35, n_clusters=5,  sharp=1.0)),
        ("clustG eps.10 nc15",       dict(structure="cluster_gauss", eps=0.10, n_clusters=15, sharp=1.0)),
        ("clustG eps.10 nc40",       dict(structure="cluster_gauss", eps=0.10, n_clusters=40, sharp=1.0)),
        ("clustG eps.10 nc5 sharp3", dict(structure="cluster_gauss", eps=0.10, n_clusters=5,  sharp=3.0)),
        ("gauss continuous",         dict(structure="gauss",         eps=0.00, n_clusters=0,  sharp=1.0)),
        ("gauss sharp3",             dict(structure="gauss",         eps=0.00, n_clusters=0,  sharp=3.0)),
    ]

    print("=" * 100)
    print("LATENT STRUCTURE SWEEP (reward-observable p=1, sample-starved). "
          "m=%d n=%d d=%d T=%d cand=%d %d seeds" % (m, n, d, T, cand, len(seeds)))
    print("skill=(greedy-rand)/(oracle-rand). CF-Tab = skill gap. range=oracle-rand (easiness).")
    print("=" * 100)
    print(f"{'structure':28s}  {'Tab skill':>9s}  {'CF skill':>9s}  {'CF-Tab':>7s}  "
          f"{'oracle':>7s}  {'rand':>6s}  {'range':>6s}")
    print("-" * 100)
    for label, cfg in configs:
        ts, _, oc, rd = skill_over_seeds(Tabular, tab_hp, cfg, m, n, d, T, cand, sigma, p, seeds)
        cs, _, _, _ = skill_over_seeds(RefitMF, cf_hp, cfg, m, n, d, T, cand, sigma, p, seeds)
        print(f"{label:28s}  {ts:9.3f}  {cs:9.3f}  {cs-ts:7.3f}  "
              f"{oc:7.3f}  {rd:6.3f}  {oc-rd:6.3f}")
    print("-" * 100)
    print("READ: large CF-Tab gap = structure where low-rank CF beats tabular most.")
    print("Watch 'range': tiny range = 'easy' structure (random already near oracle).")


if __name__ == "__main__":
    main()
