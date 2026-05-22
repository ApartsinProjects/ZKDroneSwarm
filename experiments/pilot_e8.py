"""
E8 (ACTIVE EXPLORATION confirmation): ActiveCFconv (latent-space UCB exploration +
converged ALS) vs RewardCF (eps-greedy), HybridCFconv (probe-then-fit), PTF.
unseen + anytime at rho in {1.0,0.5,0.25}, 12 seeds, paired bootstrap 95% CIs for
ActiveCFconv - RewardCF. Hypothesis: active (uncertainty-directed) exploration,
made collective via the broadcast counts, improves unseen AND keeps top anytime
(no probe-phase cost) -> dominates eps-greedy RewardCF.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_anytime as pa
from _results_io import save_results

METHODS = ["RewardCF", "HybridCFconv", "ActiveCFconv", "PTF"]
RHOS = [1.0, 0.5, 0.25]
SEEDS = list(range(12))
RNG = np.random.RandomState(0)


def _job(args):
    kind, nm, rho, s = args
    if kind == "unseen":
        return kind, nm, rho, s, pc._run_cell((nm, rho, s))[4]
    return kind, nm, rho, s, float(np.asarray(pa.run_anytime((nm, rho, s))[3])[-1])


def _ci(d, B=10000):
    a = np.asarray(d); idx = RNG.randint(0, len(a), (B, len(a))); m = a[idx].mean(1)
    return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    jobs = [(k, nm, rho, s) for k in ("unseen", "anytime") for rho in RHOS for nm in METHODS for s in SEEDS]
    raw = {k: {str(r): {nm: [None] * len(SEEDS) for nm in METHODS} for r in RHOS} for k in ("unseen", "anytime")}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            k, nm, rho, s, v = fut.result(); raw[k][str(rho)][nm][s] = v
            done += 1
            if done % 48 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))
    for k in ("unseen", "anytime"):
        print("\n[%s] mean (12 seeds) | ActiveCFconv - RewardCF paired 95%% CI" % k)
        print("%6s | " % "rho" + " ".join("%13s" % nm for nm in METHODS) + " | Active-Reward [CI]")
        for rho in RHOS:
            means = {nm: np.mean(raw[k][str(rho)][nm]) for nm in METHODS}
            d = np.array(raw[k][str(rho)]["ActiveCFconv"]) - np.array(raw[k][str(rho)]["RewardCF"])
            md, lo, hi = _ci(d); sig = "win" if lo > 0 else ("tie" if hi > 0 else "loss")
            print("%6.2f | " % rho + " ".join("%13.3f" % means[nm] for nm in METHODS)
                  + " | %+.3f [%+.3f,%+.3f] %s" % (md, lo, hi, sig))
    path = save_results("e8_active", {
        "meta": {"experiment": "E8 active exploration (ActiveCFconv) confirmation",
                 "methods": METHODS, "rhos": RHOS, "seeds": SEEDS, "m": pc.M, "n": pc.N,
                 "d": pc.D, "d_hat": pc.D_HAT, "T": pc.T, "sigma_obs": pc.SB,
                 "metric": "unseen + anytime; paired ActiveCFconv-RewardCF CI"},
        "raw": raw})
    print("saved ->", path)


if __name__ == "__main__":
    main()
