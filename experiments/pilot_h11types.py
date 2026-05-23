"""E-H11types: does de-confliction depend on DRONE-TYPE HOMOGENEITY? Sweep the number of
drone types K1 (k1=1 -> all drones share one rank-1 preference = maximal contention overlap;
k1 large -> distinct preferences). HYP: the proactive PRIVATE offset (T7) helps MOST when
drones are homogeneous (everyone wants the same targets), and the gap over greedy shrinks as
types diversify. Fixed pool (capacity-1), 8 seeds, bootstrap 95% CI. Writes docs/H11TYPES.md."""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_contention as pcon
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K1S = [1, 2, 5, 30]                 # drone-type counts: 1 = identical, 30 = all distinct
POOL = 30                          # capacity-1 contention pool (= m, so binding but not pigeonhole-saturated)
SEEDS = list(range(8))
ORDER = ["ContentionAdaCF", "CBBAlite", "MusicalChairs", "RewardCFconv"]
RNG = np.random.RandomState(0)


def _job(args):
    nm, k1, seed = args
    Cls, hp = pcon.REG[nm]
    a, u, c = pcon.run_contention(Cls, hp, 1.0, POOL, seed, k1=k1)
    return nm, k1, seed, float(a), float(c)


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v is not None], float)
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, k1, s) for k1 in K1S for nm in ORDER for s in SEEDS]
    raw = {str(k1): {nm: {"earned": [None] * len(SEEDS), "coll": [None] * len(SEEDS)}
                     for nm in ORDER} for k1 in K1S}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, k1, s, a, c = fut.result()
            raw[str(k1)][nm]["earned"][s] = a
            raw[str(k1)][nm]["coll"][s] = c
            done += 1
            if done % 16 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("h11types", {
        "meta": {"experiment": "de-confliction vs drone-type homogeneity (K1 sweep) under contention",
                 "methods": ORDER, "k1s": K1S, "pool": POOL, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT,
                 "metric": "earned-reward skill (matching-normalized) + collision rate, per K1"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# De-confliction vs drone-type homogeneity (E-H11types)\n",
         "Earned-reward skill at pool=%d (capacity-1), sweeping the number of DRONE TYPES K1 "
         "(K1=1 = all drones identical/rank-1 = maximal overlap; K1=30 = all distinct). 8 seeds, "
         "bootstrap 95%% CI.\n" % POOL,
         "| method | " + " | ".join("K1=%d" % k for k in K1S) + " |",
         "|" + "---|" * (len(K1S) + 1)]
    for nm in ORDER:
        cells = [cell(raw[str(k)][nm]["earned"]) for k in K1S]
        lab = "**%s**" % nm if "Contention" in nm else nm
        L.append("| %s | %s |" % (lab, " | ".join(cells)))
    L.append("")
    # gap (ours - greedy) per K1, to test the hypothesis directly
    L.append("Gap ContentionAdaCF - greedy RewardCFconv (de-confliction value) by K1:")
    gaps = []
    for k in K1S:
        g = float(np.mean(raw[str(k)]["ContentionAdaCF"]["earned"]) - np.mean(raw[str(k)]["RewardCFconv"]["earned"]))
        gaps.append("K1=%d: %+.3f" % (k, g))
    L.append("  " + ";  ".join(gaps) + "\n")
    L.append("Read: if the gap SHRINKS as K1 grows, de-confliction matters most under TYPE HOMOGENEITY "
             "(identical drones fight over the same targets, so proactive private spreading pays off); "
             "with diverse types drones naturally spread and the offset adds little.\n")
    out_md = os.path.join(ROOT, "docs", "H11TYPES.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
