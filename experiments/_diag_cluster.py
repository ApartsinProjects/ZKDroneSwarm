"""Test the hypothesis: are the softmax modality profiles CLUSTER-LIKE (quasi-discrete types) rather
than a continuum, and is that what favours the neighborhood/clustering baselines (kNN-CF, CLUB)?

Sweep the profile concentration `conc` (softmax temperature): conc->0 = uniform profiles (pure
continuum, no types), conc->inf = one-hot (S discrete types). For each conc measure how clustered the
profiles are (mean peak weight; between-type / total variance ratio) AND the unseen-pair skill of
kNN-CF, CLUB, SwarmCF, SwarmCF+bias. If the hypothesis holds, kNN/CLUB beat SwarmCF MORE as conc grows
(more discrete), and SwarmCF(+bias) is competitive/ahead at low conc (true continuum).

Run:  python experiments/_diag_cluster.py
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentswarm.sweeps import base_config
from emergent_lowrank import physical_reward, wide_factors, run_one, eff_rank

METHODS = ["knn_cf", "club", "swarm_cf", "swarm_cf_bias"]
PRETTY = {"knn_cf": "kNN-CF", "club": "CLUB", "swarm_cf": "SwarmCF", "swarm_cf_bias": "SwarmCF+bias"}


def profile_cluster_score(rng, count, S, conc):
    """How cluster-like are the softmax profiles? Returns (mean peak weight, between/total var ratio).
    peak weight: 1/S = uniform (continuum), 1.0 = one-hot (discrete). ratio: fraction of profile
    variance explained by grouping each profile by its DOMINANT modality (high = discrete S-types)."""
    z = rng.randn(count, S); A = np.exp(conc * z); A /= A.sum(1, keepdims=True)
    maxw = float(np.mean(A.max(1)))
    dom = A.argmax(1); g = A.mean(0)
    tot = float(((A - g) ** 2).sum())
    bet = 0.0
    for k in range(S):
        sel = dom == k
        if sel.any():
            bet += int(sel.sum()) * float(((A[sel].mean(0) - g) ** 2).sum())
    return maxw, bet / max(tot, 1e-9)


def main():
    t0 = time.time()
    seeds = list(range(8)); S = 4
    concs = [0.25, 0.5, 1.0, 1.5, 2.5, 4.0]    # 0.25 ~ near-uniform profiles (pure-geometry sensing)
    base = base_config(rho=0.25)

    print("=== Profile clustering + method skill vs concentration (S=%d, rho=0.25, %d seeds) ===" % (S, len(seeds)))
    print("conc | peakw btw/tot | rank95 rank99 | " + " ".join("%-12s" % PRETTY[m] for m in METHODS))
    rows = []
    for conc in concs:
        mw, bt, r95, r99 = [], [], [], []
        res = {m: [] for m in METHODS}
        for s in seeds:
            wr = np.random.RandomState(s)
            R = physical_reward(wr, base.m, base.n, S, conc=conc)
            P, U = wide_factors(R)
            d_guess = base.rank_for_run(wr)
            er, _ = eff_rank(R); r95.append(er[0.95]); r99.append(er[0.99])
            for m in METHODS:
                res[m].append(run_one(base, P, U, d_guess, s, m))
            cs = profile_cluster_score(np.random.RandomState(1000 + s), base.m, S, conc)
            mw.append(cs[0]); bt.append(cs[1])
        skills = {m: float(np.mean(res[m])) for m in METHODS}
        rows.append((conc, np.mean(mw), np.mean(bt), np.mean(r95), np.mean(r99), skills))
        print("%.2f | %.3f  %.3f | %5.1f  %5.1f | " % (conc, np.mean(mw), np.mean(bt), np.mean(r95), np.mean(r99))
              + " ".join("%-12.3f" % skills[m] for m in METHODS), flush=True)

    print("\n  gap (kNN - SwarmCF+bias) vs conc:  " +
          "  ".join("c=%.2f:%+.3f" % (r[0], r[5]["knn_cf"] - r[5]["swarm_cf_bias"]) for r in rows))
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
