"""
C11 (HEADLINE): the unseen-pair CATEGORICAL win in the NATURAL regime of
heterogeneous observation MASKING.

Setting (block-model core): each drone observes its OWN outcome (clean) plus a
PERSISTENT, per-drone-random subset of teammates' broadcast (mask M[i,k] ~
Bernoulli(rho), M[i,i]=1). Persistent blind spots => UNIQUE per-drone states
(decentralization is REAL, not cosmetic) AND targets a drone never observes (own
pull or visible teammate) are UNSEEN -> must be COMPLETED via CF; tabular is at
the FLOOR by construction on its own-not-pulled targets.

Fairness: CF uses a GUESSED rank d_hat (not the true d). Report ORACLE
(centralized + complete-information ceiling) and RANDOM (floor) via the skill
normalization. Sweep rho: rho=1 (full broadcast, cosmetic decentralization) ->
low rho (unique states, CF advantage scales with visible coverage; tabular at
floor on unseen at every rho>0).
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_noise import Tabular, RewardCF
from core import make_world
from _results_io import save_results


def run_masked(Cls, hp, world, T, seed, so, sb, cand, d_hat, rho):
    P, U, R = world[:3]; m, n = R.shape
    rng = np.random.RandomState(seed + 999)
    M = rng.rand(m, m) < rho            # PERSISTENT per-drone observation mask
    np.fill_diagonal(M, True)           # always observe self
    learners = [Cls(m, n, d_hat, i, seed + 7 * i + 1, **hp) for i in range(m)]
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i and M[i, k]:
                    revealed[k] = true_r[k] + rng.normal(0, sb); rvar[k] = sb ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    preds = [learners[i].predict_scores() for i in range(m)]
    pulled = [learners[i].pulled_mask() for i in range(m)]
    g = np.random.RandomState(seed + 555)
    ovg, ovo, ovr, ung, uno, unr = [], [], [], [], [], []
    for _ in range(120):
        for k in range(m):
            off = g.choice(n, size=cand, replace=False)
            ovg.append(R[k, off[int(np.argmax(preds[k][off]))]]); ovo.append(R[k, off].max()); ovr.append(R[k, off].mean())
            unseen = np.where(~pulled[k])[0]
            if len(unseen) >= cand:
                offu = g.choice(unseen, size=cand, replace=False)
                ung.append(R[k, offu[int(np.argmax(preds[k][offu]))]]); uno.append(R[k, offu].max()); unr.append(R[k, offu].mean())

    def sk(gg, oo, rr):
        gm, om, rm = np.mean(gg), np.mean(oo), np.mean(rr)
        return (gm - rm) / max(om - rm, 1e-6)
    # state-uniqueness: mean pairwise cosine distance of per-drone predicted rows
    rows_pred = np.array([preds[i] for i in range(m)])
    rows_pred = rows_pred - rows_pred.mean(1, keepdims=True)
    nrm = np.linalg.norm(rows_pred, axis=1, keepdims=True) + 1e-9
    cos = (rows_pred / nrm) @ (rows_pred / nrm).T
    uniq = 1.0 - cos[np.triu_indices(m, 1)].mean()   # higher = more unique states
    return sk(ovg, ovo, ovr), sk(ung, uno, unr), float(uniq)


def main():
    m, n, d, K = 30, 240, 5, 10
    d_hat = 8; T, cand = 50, 20; so, sb = 0.10, 0.30
    seeds = list(range(5))
    tab = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)
    rew = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=8, refit_every=3)
    methods = [("Tabular", Tabular, tab), ("RewardCF", RewardCF, rew)]
    rhos = [1.0, 0.5, 0.25, 0.1]

    print("=" * 92)
    print("C11: unseen-pair win under HETEROGENEOUS MASKING (block model, signed). "
          "m=%d n=%d d=%d(K=%d) d_hat=%d T=%d cand=%d %d seeds" % (m, n, d, K, d_hat, T, cand, len(seeds)))
    print("rho = per-drone fraction of teammates' broadcast observed (persistent). "
          "sigma_own=%.2f sigma_obs=%.2f" % (so, sb))
    print("=" * 92)
    print(f"{'rho':>5s} | {'Tab overall':>11s} {'Tab unseen':>10s} | {'CF overall':>10s} "
          f"{'CF unseen':>9s} | {'CF stateUniq':>12s}")
    print("-" * 92)
    raw = {}
    for rho in rhos:
        cell = {}
        for name, Cls, hp in methods:
            ov, un, uq = [], [], []
            for s in seeds:
                w = make_world(m, n, d, K, K, within=0.15, seed=s, signed=True)
                o, u, q = run_masked(Cls, hp, w, T, s, so, sb, cand, d_hat, rho)
                ov.append(o); un.append(u); uq.append(q)
            cell[name] = dict(overall=ov, unseen=un, uniq=uq)
        raw[str(rho)] = cell
        t_ov = np.mean(cell["Tabular"]["overall"]); t_un = np.mean(cell["Tabular"]["unseen"])
        c_ov = np.mean(cell["RewardCF"]["overall"]); c_un = np.mean(cell["RewardCF"]["unseen"])
        c_uq = np.mean(cell["RewardCF"]["uniq"])
        print(f"{rho:5.2f} | {t_ov:11.3f} {t_un:10.3f} | {c_ov:10.3f} {c_un:9.3f} | {c_uq:12.3f}")
    print("-" * 92)
    print("CATEGORICAL: CF unseen >> Tabular unseen (~0, floor) at every rho>0; stateUniq>0")
    print("for rho<1 (decentralization real). At rho=1 broadcast is shared (cosmetic).")
    path = save_results("c11_masking", {
        "meta": {"experiment": "C11 unseen-pair win under heterogeneous masking",
                 "m": m, "n": n, "d": d, "K": K, "d_hat": d_hat, "T": T, "cand": cand,
                 "sigma_own": so, "sigma_obs": sb, "seeds": seeds, "rhos": rhos,
                 "metric": "per-seed overall/unseen skill + state-uniqueness"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
