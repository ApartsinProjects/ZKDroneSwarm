"""
E2/E4/E6 (SCALING SWEEPS): how the unseen and anytime edges scale with the
parameters we held fixed. One parametric runner, one knob at a time, others at
defaults (rho=0.5 masked regime, sigma_obs=0.30).
  E2  true rank d        in {1,2,3,5,8,12,20}   (gap scales with low-rankness)
  E4a horizon T          in {25,50,100,200}      (n>>T starvation -> anytime edge)
  E4b targets n          in {120,240,480,960}    (more starvation)
  E6  guessed rank d_hat in {2,3,5,8,12,20}      (robustness to misguess; fairness)
Methods: Tabular, UCBIndep (no-structure), PTF (batch-SVD competitor), RewardCF,
HybridCF (ours). Metrics: final-policy unseen + anytime cumulative.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from core import make_world
from _results_io import save_results

METHODS = ["Tabular", "UCBIndep", "PTF", "RewardCF", "HybridCF"]
SWEEPS = {
    "d":    [1, 2, 3, 5, 8, 12, 20],
    "T":    [25, 50, 100, 200],
    "n":    [120, 240, 480, 960],
    "dhat": [2, 3, 5, 8, 12, 20],
}
SEEDS = list(range(8))
RHO, SIG = 0.5, 0.30


def run_param(args):
    name, sweep, val, seed = args
    d, n, T, d_hat = 5, pc.N, pc.T, pc.D_HAT
    if sweep == "d": d = val
    elif sweep == "n": n = val
    elif sweep == "T": T = val
    elif sweep == "dhat": d_hat = val
    Cls, base = pc.REGISTRY[name]; hp = dict(base)
    if "T_total" in hp: hp["T_total"] = T            # keep probe/explore budget aligned with T
    w = make_world(pc.M, n, d, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m = pc.M; cand, so = pc.CAND, pc.SO
    rng = np.random.RandomState(seed + 999)
    Mask = rng.rand(m, m) < RHO; np.fill_diagonal(Mask, True)
    learners = [Cls(m, n, d_hat, i, seed + 7 * i + 1, **hp) for i in range(m)]
    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        real[t] = true_r.sum()
        orac[t] = sum(R[i, cand_sets[i]].max() for i in range(m))
        rnd[t] = sum(R[i, cand_sets[i]].mean() for i in range(m))
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i and Mask[i, k]:
                    revealed[k] = true_r[k] + rng.normal(0, SIG); rvar[k] = SIG ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    preds = [learners[i].predict_scores() for i in range(m)]
    pulled = [learners[i].pulled_mask() for i in range(m)]
    g = np.random.RandomState(seed + 555)
    ung, uno, unr = [], [], []
    for _ in range(80):
        for k in range(m):
            unseen = np.where(~pulled[k])[0]
            if len(unseen) >= cand:
                offu = g.choice(unseen, size=cand, replace=False)
                ung.append(R[k, offu[int(np.argmax(preds[k][offu]))]]); uno.append(R[k, offu].max()); unr.append(R[k, offu].mean())
    gm, om, rm = np.mean(ung), np.mean(uno), np.mean(unr)
    unseen_sk = float((gm - rm) / max(om - rm, 1e-6))
    cr, co, cd = np.cumsum(real), np.cumsum(orac), np.cumsum(rnd)
    anytime = float((cr[-1] - cd[-1]) / max(co[-1] - cd[-1], 1e-9))
    return name, sweep, val, seed, unseen_sk, anytime


def main():
    print("=" * 92)
    print("E2/E4/E6 SCALING SWEEPS (rho=%.2f sigma_obs=%.2f, %d seeds). unseen + anytime."
          % (RHO, SIG, len(SEEDS)))
    print("=" * 92)
    jobs = [(nm, sw, v, s) for sw, vals in SWEEPS.items() for v in vals for nm in METHODS for s in SEEDS]
    raw = {sw: {str(v): {nm: {"unseen": [], "anytime": []} for nm in METHODS} for v in vals}
           for sw, vals in SWEEPS.items()}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_param, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, sw, v, s, un, an = fut.result()
            c = raw[sw][str(v)][nm]; c["unseen"].append(un); c["anytime"].append(an)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print("  ... %d/%d cells done" % (done, len(jobs)))

    for sw, vals in SWEEPS.items():
        for metric in ["unseen", "anytime"]:
            print("\n[%s] %s skill (mean/%d seeds)" % (sw, metric, len(SEEDS)))
            print("%9s | " % sw + " ".join("%7s" % str(v) for v in vals))
            for nm in METHODS:
                row = [np.mean(raw[sw][str(v)][nm][metric]) for v in vals]
                print("%9s | " % nm + " ".join("%7.3f" % x for x in row))

    path = save_results("e246_scaling", {
        "meta": {"experiment": "E2/E4/E6 scaling sweeps (d, T, n, d_hat)",
                 "methods": METHODS, "sweeps": SWEEPS, "seeds": SEEDS,
                 "rho": RHO, "sigma_obs": SIG, "m": pc.M, "K": pc.K,
                 "defaults": {"d": 5, "n": pc.N, "T": pc.T, "d_hat": pc.D_HAT},
                 "metric": "per-seed unseen + anytime skill vs each swept param"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
