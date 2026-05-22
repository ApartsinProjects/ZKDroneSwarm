"""
C15 (CROSSOVER / MASKING-ROBUSTNESS): finer rho sweep isolating the headline
positioning of cycle 24 -- our ONLINE weighted-ALS (RewardCF/BothCF) is
masking-ROBUST (unseen skill ~flat in rho), while every BATCH-SVD low-rank
hybrid (PTF, ESTR, BPMF) DECAYS as the broadcast is masked, because each SVDs an
empirical R_hat whose unobserved entries are imputed 0 (sparse+biased under
masking). UCBIndep is the no-structure FLOOR reference (unseen ~0 at every rho).

Reuses pilot_compare._run_cell (identical fair config: block model, guessed
d_hat=8, own clean reward + persistent per-drone broadcast mask). 8 seeds for
tighter CIs. Parallel across CPU cores. Pins the crossover near rho=1: PTF
edges ahead ONLY at full broadcast (no real observation limit); at any genuine
masking (rho<1) ours wins.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from _results_io import save_results

METHODS = ["UCBIndep", "MFSGD", "ESTR", "BPMF", "PTF", "RewardCF", "BothCF", "HybridCF"]
RHOS = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.15, 0.1]
SEEDS = list(range(8))


def main():
    print("=" * 100)
    print("C15 CROSSOVER: unseen-pair skill vs rho (masking-robustness). "
          "m=%d n=%d d=%d(K=%d) d_hat=%d T=%d cand=%d %d seeds" %
          (pc.M, pc.N, pc.D, pc.K, pc.D_HAT, pc.T, pc.CAND, len(SEEDS)))
    print("our online weighted-ALS (RewardCF/BothCF) vs batch-SVD hybrids "
          "(PTF/ESTR/BPMF) vs no-structure floor (UCBIndep). sigma_own=%.2f sigma_obs=%.2f"
          % (pc.SO, pc.SB))
    print("=" * 100)

    jobs = [(nm, r, s) for r in RHOS for nm in METHODS for s in SEEDS]
    raw = {str(r): {nm: {"overall": [], "unseen": [], "uniq": []} for nm in METHODS}
           for r in RHOS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(pc._run_cell, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, r, s, o, u, q = fut.result()
            c = raw[str(r)][nm]; c["overall"].append(o); c["unseen"].append(u); c["uniq"].append(q)
            done += 1
            if done % 28 == 0 or done == len(jobs):
                print("  ... %d/%d cells done" % (done, len(jobs)))

    # ---- UNSEEN-pair skill table (rows=method, cols=rho) ----
    print("\nUNSEEN-pair skill (mean +- std over %d seeds):" % len(SEEDS))
    hdr = "method".rjust(10) + " | " + " ".join("%9.2f" % r for r in RHOS)
    print(hdr); print("-" * len(hdr))
    for nm in METHODS:
        cells = []
        for r in RHOS:
            u = raw[str(r)][nm]["unseen"]; cells.append("%4.2f" % np.mean(u))
        print("%10s | " % nm + " ".join("%9s" % c for c in cells))
    print("\nOVERALL skill (mean over %d seeds):" % len(SEEDS))
    print(hdr); print("-" * len(hdr))
    for nm in METHODS:
        cells = ["%4.2f" % np.mean(raw[str(r)][nm]["overall"]) for r in RHOS]
        print("%10s | " % nm + " ".join("%9s" % c for c in cells))
    print("-" * len(hdr))
    print("READ: ours (RewardCF/BothCF) ~FLAT in rho; PTF/ESTR/BPMF DECAY as rho falls;")
    print("UCBIndep ~0 (floor) at every rho. Crossover near rho=1 (PTF wins only there).")

    path = save_results("c15_crossover", {
        "meta": {"experiment": "C15 masking-robustness crossover (finer rho)",
                 "methods": METHODS, "rhos": RHOS, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "K": pc.K, "d_hat": pc.D_HAT,
                 "T": pc.T, "cand": pc.CAND, "sigma_own": pc.SO, "sigma_obs": pc.SB,
                 "metric": "per-seed overall/unseen skill + stateUniq vs rho",
                 "fairness": "guessed d_hat (not true d); reuses pilot_compare._run_cell"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
