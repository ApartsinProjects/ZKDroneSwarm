"""
C13: unseen-pair generalization vs TRUE RANK d -- empirical complement to the D3
theory (CF completes a rank-d row from ~O(d) observations; tabular has zero info
on unseen pairs at any d).

Reward-observable (full broadcast), sample-starved, block-model core. CF uses a
FIXED guessed rank d_hat (fair). Sweep the TRUE rank d. Expect: CF unseen-pair
skill DECREASES as d rises (more parameters to complete from the same data);
Tabular stays at the FLOOR (~0) at every d. This grounds the Theta(d)-vs-Theta(n)
separation: the CF advantage on unseen pairs scales with low-rankness.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_noise import Tabular, RewardCF
from core import make_world
from _results_io import save_results


def train_eval(Cls, hp, world, T, seed, so, sb, cand, d_hat):
    P, U, R = world[:3]; m, n = R.shape
    rng = np.random.RandomState(seed + 999)
    learners = [Cls(m, n, d_hat, i, seed + 7 * i + 1, **hp) for i in range(m)]
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):                       # p=1 full broadcast (reward-observable)
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i:
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
            uns = np.where(~pulled[k])[0]
            if len(uns) >= cand:
                offu = g.choice(uns, size=cand, replace=False)
                ung.append(R[k, offu[int(np.argmax(preds[k][offu]))]]); uno.append(R[k, offu].max()); unr.append(R[k, offu].mean())

    def sk(gg, oo, rr):
        gm, om, rm = np.mean(gg), np.mean(oo), np.mean(rr)
        return (gm - rm) / max(om - rm, 1e-6)
    return sk(ovg, ovo, ovr), sk(ung, uno, unr)


def main():
    m, n, K = 30, 240, 20
    d_hat = 10; T, cand = 50, 20; so, sb = 0.10, 0.10
    seeds = list(range(5))
    ranks = [2, 3, 5, 8]
    tab = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)
    rew = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=8, refit_every=3)
    print("=" * 80)
    print("C13: unseen-pair skill vs TRUE RANK d (D3 support). m=%d n=%d K=%d d_hat=%d "
          "T=%d %d seeds, reward-observable" % (m, n, K, d_hat, T, len(seeds)))
    print("=" * 80)
    print(f"{'true d':>6s} | {'Tab unseen':>10s} | {'CF unseen':>9s} | {'CF overall':>10s} | {'CF-Tab unseen':>13s}")
    print("-" * 80)
    raw = {}
    for d in ranks:
        tu, cu, co = [], [], []
        for s in seeds:
            w = make_world(m, n, d, K, K, within=0.15, seed=s, signed=True)
            _, t_un = train_eval(Tabular, tab, w, T, s, so, sb, cand, d_hat)
            c_ov, c_un = train_eval(RewardCF, rew, w, T, s, so, sb, cand, d_hat)
            tu.append(t_un); cu.append(c_un); co.append(c_ov)
        raw[str(d)] = dict(tab_unseen=tu, cf_unseen=cu, cf_overall=co)
        print(f"{d:6d} | {np.mean(tu):10.3f} | {np.mean(cu):9.3f} | {np.mean(co):10.3f} | "
              f"{np.mean(cu)-np.mean(tu):+13.3f}")
    print("-" * 80)
    print("EXPECT: CF unseen-pair skill DECREASES with true d (more to complete from same")
    print("data); Tabular ~0 (floor) at all d. Grounds the O(d) completion / Theta(d)-vs-n story.")
    path = save_results("c13_rank_unseen", {
        "meta": {"experiment": "C13 unseen-pair skill vs true rank", "m": m, "n": n,
                 "K": K, "d_hat": d_hat, "T": T, "cand": cand, "sigma_own": so,
                 "sigma_obs": sb, "seeds": seeds, "ranks": ranks,
                 "metric": "per-seed Tabular/CF unseen skill + CF overall vs true d"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
