"""
Wave 3b-1 (EXTERNAL VALIDITY to the generative assumptions): does the advantage
survive when the world is NOT exactly low-rank? We stress two assumptions:
  nonlin : apply a power link  R -> sign(R)|R|^(1/(1+nonlin))  (identity at 0), which
           RAISES the effective rank (a rank-d matrix put through an elementwise
           nonlinearity is generally higher rank).
  approx : add entrywise Gaussian noise to R (clip to [-1,1]) so the truth is only
           APPROXIMATELY low-rank.
Methods still use the GUESSED rank d_hat. We sweep each knob at rho=0.5 and report
unseen + anytime skill and the realized effective rank. Expect graceful degradation
and our methods still beating the baselines.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from core import make_world, effective_rank
from _results_io import save_results

METHODS = ["UCBIndep", "PTF", "RewardCF", "HybridCFconv", "ActiveCFconv"]
SWEEPS = {"nonlin": [0.0, 0.5, 1.0, 2.0], "approx": [0.0, 0.1, 0.2, 0.4]}
SEEDS = list(range(8))
RHO, SOBS, SOWN, CAND = 0.5, 0.30, 0.10, 20


def transform_R(R, nonlin, approx, rng):
    X = R.copy()
    if nonlin > 0:
        X = np.sign(X) * np.abs(X) ** (1.0 / (1.0 + nonlin))   # power link, identity at 0
    if approx > 0:
        X = np.clip(X + approx * rng.randn(*R.shape), -1.0, 1.0)
    return X


def run_stress(args):
    name, knob, val, seed = args
    nonlin = val if knob == "nonlin" else 0.0
    approx = val if knob == "approx" else 0.0
    Cls, base = pc.REGISTRY[name]; hp = dict(base)
    if "T_total" in hp: hp["T_total"] = pc.T
    P, U, R0, meta = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    R = transform_R(R0, nonlin, approx, np.random.RandomState(seed + 12345))
    er = float(effective_rank(R))
    m, n, T = pc.M, pc.N, pc.T
    rng = np.random.RandomState(seed + 999)
    Mask = rng.rand(m, m) < RHO; np.fill_diagonal(Mask, True)
    learners = [Cls(m, n, pc.D_HAT, i, seed + 7 * i + 1, **hp) for i in range(m)]
    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=CAND, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        real[t] = true_r.sum(); orac[t] = sum(R[i, cand_sets[i]].max() for i in range(m))
        rnd[t] = sum(R[i, cand_sets[i]].mean() for i in range(m))
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, SOWN); rvar[i] = SOWN ** 2
            for k in range(m):
                if k != i and Mask[i, k]:
                    revealed[k] = true_r[k] + rng.normal(0, SOBS); rvar[k] = SOBS ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    preds = [learners[i].predict_scores() for i in range(m)]
    pulled = [learners[i].pulled_mask() for i in range(m)]
    g = np.random.RandomState(seed + 555); ung, uno, unr = [], [], []
    for _ in range(80):
        for k in range(m):
            uns = np.where(~pulled[k])[0]
            if len(uns) >= CAND:
                off = g.choice(uns, size=CAND, replace=False)
                ung.append(R[k, off[int(np.argmax(preds[k][off]))]]); uno.append(R[k, off].max()); unr.append(R[k, off].mean())
    unseen = float((np.mean(ung) - np.mean(unr)) / max(np.mean(uno) - np.mean(unr), 1e-6))
    cr, co, cd = np.cumsum(real), np.cumsum(orac), np.cumsum(rnd)
    anytime = float((cr[-1] - cd[-1]) / max(co[-1] - cd[-1], 1e-9))
    return name, knob, val, seed, unseen, anytime, er


def main():
    print("=" * 88)
    print("Wave 3b-1 STRESS: approximate-low-rank + nonlinear link (rho=%.2f, %d seeds)" % (RHO, len(SEEDS)))
    print("=" * 88)
    jobs = [(nm, kn, v, s) for kn, vals in SWEEPS.items() for v in vals for nm in METHODS for s in SEEDS]
    raw = {kn: {str(v): {nm: {"unseen": [], "anytime": [], "er": []} for nm in METHODS} for v in vals}
           for kn, vals in SWEEPS.items()}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_stress, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, kn, v, s, un, an, er = fut.result()
            c = raw[kn][str(v)][nm]; c["unseen"].append(un); c["anytime"].append(an); c["er"].append(er)
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))
    for kn, vals in SWEEPS.items():
        ers = [np.mean(raw[kn][str(v)][METHODS[0]]["er"]) for v in vals]
        print("\n[%s] effective rank: %s" % (kn, ["%.0f" % e for e in ers]))
        for metric in ["unseen", "anytime"]:
            print(" %s skill:" % metric)
            print("   %12s | " % kn + " ".join("%6s" % str(v) for v in vals))
            for nm in METHODS:
                print("   %12s | " % nm + " ".join("%6.3f" % np.mean(raw[kn][str(v)][nm][metric]) for v in vals))
    path = save_results("stress_assump", {
        "meta": {"experiment": "Wave3b-1 assumption stress (approx low-rank + nonlinear link)",
                 "methods": METHODS, "sweeps": SWEEPS, "seeds": SEEDS, "rho": RHO,
                 "m": pc.M, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT, "T": pc.T, "sigma_obs": SOBS,
                 "metric": "unseen + anytime + realized effective rank vs nonlin/approx"},
        "raw": raw})
    print("saved ->", path)


if __name__ == "__main__":
    main()
