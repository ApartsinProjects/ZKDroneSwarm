"""
E15 (MORE LOW-RANK / STRUCTURED BASELINES, fair, same limitations): add three
distinct alternatives, all run per-drone over the same masked broadcast with a
guessed rank where applicable -> SoftImpute (nuclear-norm convex completion), KNNCF
(memory-based, model-free CF), BiasModel (additive drone+target bias, no
interaction). Compare vs PTF and our RewardCF / HybridCFconv / ActiveCFconv on
unseen + anytime across rho, 8 seeds, with paired bootstrap CIs (ours vs the best
new baseline). Tests: does any alternative match us, especially under masking?
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_anytime as pa
from _results_io import save_results

NEW = ["BiasModel", "KNNCF", "SoftImpute"]
OURS = ["RewardCF", "HybridCFconv", "ActiveCFconv"]
METHODS = NEW + ["PTF"] + OURS
RHOS = [1.0, 0.5, 0.25, 0.1]
SEEDS = list(range(8))
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
    print("=" * 100)
    print("E15 MORE BASELINES (fair, same limitations): %s vs PTF vs ours. %d seeds"
          % (NEW, len(SEEDS)))
    print("=" * 100)
    jobs = [(k, nm, rho, s) for k in ("unseen", "anytime") for rho in RHOS for nm in METHODS for s in SEEDS]
    raw = {k: {str(r): {nm: [None] * len(SEEDS) for nm in METHODS} for r in RHOS} for k in ("unseen", "anytime")}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            k, nm, rho, s, v = fut.result(); raw[k][str(rho)][nm][s] = v
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))
    for k in ("unseen", "anytime"):
        print("\n[%s] mean (%d seeds)" % (k, len(SEEDS)))
        print("%6s | " % "rho" + " ".join("%12s" % nm for nm in METHODS))
        for rho in RHOS:
            print("%6.2f | " % rho + " ".join("%12.3f" % np.mean(raw[k][str(rho)][nm]) for nm in METHODS))
        # ours-best vs new-best paired margin at rho=0.25 (the masked regime)
        ob = max(OURS, key=lambda nm: np.mean(raw[k]["0.25"][nm]))
        nb = max(NEW, key=lambda nm: np.mean(raw[k]["0.25"][nm]))
        d = np.array(raw[k]["0.25"][ob]) - np.array(raw[k]["0.25"][nb])
        md, lo, hi = _ci(d); sig = "win" if lo > 0 else ("tie" if hi > 0 else "loss")
        print("  @rho=0.25 best-ours(%s) - best-new(%s): %+.3f [%+.3f,%+.3f] %s" % (ob, nb, md, lo, hi, sig))
    path = save_results("e15_morebase", {
        "meta": {"experiment": "E15 more low-rank/structured baselines (fair)",
                 "methods": METHODS, "new": NEW, "ours": OURS, "rhos": RHOS, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT, "T": pc.T, "sigma_obs": pc.SB,
                 "metric": "unseen + anytime vs rho; paired ours-vs-new CI at rho=0.25"},
        "raw": raw})
    print("saved ->", path)


if __name__ == "__main__":
    main()
