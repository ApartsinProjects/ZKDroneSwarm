"""
E14 (GENERALITY): do the conclusions depend on the latent SAMPLING / structure and
the population sizes? Sweep the axes not covered by E2/E4/E6 (which did d, T, n,
d_hat):
  m       drones in {10,20,30,60,120}      (population size; m/n ratio)
  K       clusters K1=K2 in {2,5,10,20,30} (latent granularity; K=m -> every drone
          its OWN type = no clustering = "uniform" latents; small K = tight blocks)
  within  latent spread in {0.05,..,1.0}   (tight clusters -> diffuse/uniform-ish)
Together K and within span clustered-vs-uniform latent sampling. Fixed rho=0.5,
sigma_obs=0.30. Methods: Tabular, UCBIndep (no-structure) + PTF, RewardCF, HybridCF
(low-rank). Metrics: unseen + anytime. Goal: the categorical (CF >> floor on unseen)
and anytime dominance hold across all of these.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from core import make_world
from _results_io import save_results

METHODS = ["Tabular", "UCBIndep", "PTF", "RewardCF", "HybridCF"]
SWEEPS = {"m": [10, 20, 30, 60, 120], "K": [2, 5, 10, 20, 30],
          "within": [0.05, 0.15, 0.3, 0.6, 1.0]}
SEEDS = list(range(8))
RHO, SIG = 0.5, 0.30


def run_robust(args):
    name, sweep, val, seed = args
    m, n, d, K, within, d_hat, T = pc.M, pc.N, pc.D, pc.K, 0.15, pc.D_HAT, pc.T
    if sweep == "m": m = val
    elif sweep == "K": K = val
    elif sweep == "within": within = val
    Cls, base = pc.REGISTRY[name]; hp = dict(base)
    if "T_total" in hp: hp["T_total"] = T
    P, U, R, meta = make_world(m, n, d, K, K, within=within, seed=seed, signed=True)
    cand, so = pc.CAND, pc.SO
    rng = np.random.RandomState(seed + 999)
    Mask = rng.rand(m, m) < RHO; np.fill_diagonal(Mask, True)
    learners = [Cls(m, n, d_hat, i, seed + 7 * i + 1, **hp) for i in range(m)]
    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        real[t] = true_r.sum(); orac[t] = sum(R[i, cand_sets[i]].max() for i in range(m))
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
    g = np.random.RandomState(seed + 555); ung, uno, unr = [], [], []
    for _ in range(80):
        for k in range(m):
            uns = np.where(~pulled[k])[0]
            if len(uns) >= cand:
                off = g.choice(uns, size=cand, replace=False)
                ung.append(R[k, off[int(np.argmax(preds[k][off]))]]); uno.append(R[k, off].max()); unr.append(R[k, off].mean())
    unseen = float((np.mean(ung) - np.mean(unr)) / max(np.mean(uno) - np.mean(unr), 1e-6))
    cr, co, cd = np.cumsum(real), np.cumsum(orac), np.cumsum(rnd)
    anytime = float((cr[-1] - cd[-1]) / max(co[-1] - cd[-1], 1e-9))
    return name, sweep, val, seed, unseen, anytime


def main():
    print("=" * 84)
    print("E14 GENERALITY: m / K / within sweeps (rho=%.2f sigma=%.2f, %d seeds)"
          % (RHO, SIG, len(SEEDS)))
    print("=" * 84)
    jobs = [(nm, sw, v, s) for sw, vals in SWEEPS.items() for v in vals for nm in METHODS for s in SEEDS]
    raw = {sw: {str(v): {nm: {"unseen": [], "anytime": []} for nm in METHODS} for v in vals}
           for sw, vals in SWEEPS.items()}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_robust, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, sw, v, s, un, an = fut.result()
            c = raw[sw][str(v)][nm]; c["unseen"].append(un); c["anytime"].append(an)
            done += 1
            if done % 60 == 0 or done == len(jobs):
                print("  ... %d/%d cells done" % (done, len(jobs)))
    for sw, vals in SWEEPS.items():
        for metric in ["unseen", "anytime"]:
            print("\n[%s] %s (mean/%d seeds)" % (sw, metric, len(SEEDS)))
            print("%9s | " % sw + " ".join("%7s" % str(v) for v in vals))
            for nm in METHODS:
                print("%9s | " % nm + " ".join("%7.3f" % np.mean(raw[sw][str(v)][nm][metric]) for v in vals))
    path = save_results("e14_robust", {
        "meta": {"experiment": "E14 generality (m, K, within sweeps)",
                 "methods": METHODS, "sweeps": SWEEPS, "seeds": SEEDS, "rho": RHO,
                 "sigma_obs": SIG, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT, "T": pc.T,
                 "metric": "unseen + anytime vs m / K / within"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
