"""
E10 (PRECISION-GATED FUSION): does gating the CHOICE channel by per-target reward
PRECISION (BothCFPrec) make the fusion >= max(reward, choice) channel across the
noise grid? The earlier count-based gate (BothCFGated) mis-fired under noise. Reuse
the consistent both-channel masking sweep (pilot_e3_channels.run_2ch) over sigma_obs
at a couple rho. Report OVERALL skill and the dominance margin
min(BothCFPrec - RewardCF, BothCFPrec - ChoiceCF) (>=0 means strictly dominant).
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_e3_channels import run_2ch
from _results_io import save_results

METHODS = ["RewardCF", "ChoiceCF", "BothCF", "BothCFPrec"]
RHOS = [1.0, 0.5]
SIGS = [0.3, 0.6, 1.0, 2.0]
SEEDS = list(range(8))


def main():
    print("=" * 88)
    print("E10 PRECISION-GATED FUSION (overall skill vs sigma_obs, %d seeds)" % len(SEEDS))
    print("=" * 88)
    jobs = [(nm, rho, sig, s) for rho in RHOS for sig in SIGS for nm in METHODS for s in SEEDS]
    raw = {str(rho): {str(sig): {nm: [] for nm in METHODS} for sig in SIGS} for rho in RHOS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_2ch, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, rho, sig, s, ov, un, an = fut.result()
            raw[str(rho)][str(sig)][nm].append(ov)
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print("  ... %d/%d cells done" % (done, len(jobs)))
    for rho in RHOS:
        print("\nrho=%.2f  OVERALL skill vs sigma_obs" % rho)
        print("%11s | " % "sigma_obs" + " ".join("%7.2f" % s for s in SIGS))
        for nm in METHODS:
            print("%11s | " % nm + " ".join("%7.3f" % np.mean(raw[str(rho)][str(sig)][nm]) for sig in SIGS))
        dom = []
        for sig in SIGS:
            bp = np.mean(raw[str(rho)][str(sig)]["BothCFPrec"])
            rw = np.mean(raw[str(rho)][str(sig)]["RewardCF"])
            ch = np.mean(raw[str(rho)][str(sig)]["ChoiceCF"])
            dom.append(min(bp - rw, bp - ch))
        print("%11s | " % "dom-margin" + " ".join("%+7.3f" % x for x in dom) + "  (>=0 = dominates both)")
    print("-" * 88)
    print("GOAL: BothCFPrec dom-margin >= 0 across sigma_obs (gate keeps choice where rewards")
    print("are noisy/sparse, suppresses it where rewards are precise).")
    path = save_results("e10_precgate", {
        "meta": {"experiment": "E10 precision-gated fusion vs channel grid",
                 "methods": METHODS, "rhos": RHOS, "sigmas": SIGS, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d_hat": pc.D_HAT, "T": pc.T, "sigma_own": pc.SO,
                 "metric": "overall skill; dominance margin vs reward/choice channels"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
