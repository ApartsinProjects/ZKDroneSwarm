"""
E13 (CHOICE-ONLY ABLATION + ZK check): isolate the value of the ACTION (choice)
channel, and verify it survives the STRICT-ZK relaxation (global negatives, no
offered-menu observation). Consistent both-channel masking (reuses pilot_iid.run_iid,
persistent mode, sigma_obs=0.30). Sweep rho. Metrics: unseen + anytime.

Methods:
  Tabular   : own outcome only (no cross-agent)   -> ablation baseline.
  ChoiceCF  : own reward + teammates' CHOICES, exposure-debiased via offered menu.
  ChoiceZK  : same but GLOBAL negatives (within=False) -> strictly ZK (needs only
              the observed chosen action, not the menu).
  RewardCF  : own + teammates' REWARDS (reward channel reference, strictly ZK).
  BothCF    : fuse reward + choice.
Reads: (1) ChoiceCF/ChoiceZK > Tabular => the choice channel alone adds value;
(2) ChoiceZK ~ ChoiceCF => the benefit is NOT an artifact of menu observation
(strict-ZK); (3) compare choice vs reward channel at sigma_obs=0.30.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_iid import run_iid
from _results_io import save_results

METHODS = ["Tabular", "ChoiceCF", "ChoiceZK", "RewardCF", "BothCF"]
RHOS = [1.0, 0.5, 0.25, 0.1]
SEEDS = list(range(8))


def main():
    print("=" * 88)
    print("E13 CHOICE-ONLY ABLATION + ZK check (persistent mask, sigma_obs=%.2f, %d seeds)"
          % (pc.SB, len(SEEDS)))
    print("=" * 88)
    jobs = [(nm, "persistent", rho, s, pc.T) for rho in RHOS for nm in METHODS for s in SEEDS]
    raw = {str(rho): {nm: {"unseen": [], "anytime": []} for nm in METHODS} for rho in RHOS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_iid, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, mode, rho, s, T, un, an, uq = fut.result()
            c = raw[str(rho)][nm]; c["unseen"].append(un); c["anytime"].append(an)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print("  ... %d/%d cells done" % (done, len(jobs)))

    for metric in ["unseen", "anytime"]:
        print("\n[%s] skill vs rho (mean/%d seeds)" % (metric, len(SEEDS)))
        print("%9s | " % "rho" + " ".join("%7.2f" % r for r in RHOS))
        for nm in METHODS:
            row = [np.mean(raw[str(r)][nm][metric]) for r in RHOS]
            print("%9s | " % nm + " ".join("%7.3f" % x for x in row))
        # choice channel marginal (ChoiceCF - Tabular) and ZK gap (ChoiceZK - ChoiceCF)
        ch_marg = [np.mean(raw[str(r)]["ChoiceCF"][metric]) - np.mean(raw[str(r)]["Tabular"][metric]) for r in RHOS]
        zk_gap = [np.mean(raw[str(r)]["ChoiceZK"][metric]) - np.mean(raw[str(r)]["ChoiceCF"][metric]) for r in RHOS]
        print("%9s | " % "Ch-Tab" + " ".join("%+7.3f" % x for x in ch_marg) + "   (choice channel value)")
        print("%9s | " % "ZK-menu" + " ".join("%+7.3f" % x for x in zk_gap) + "   (strict-ZK vs menu; ~0 = ZK-robust)")

    path = save_results("e13_choice", {
        "meta": {"experiment": "E13 choice-only ablation + strict-ZK check",
                 "methods": METHODS, "rhos": RHOS, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "K": pc.K, "d_hat": pc.D_HAT,
                 "T": pc.T, "sigma_obs": pc.SB, "sigma_own": pc.SO,
                 "metric": "unseen + anytime skill vs rho; ChoiceZK = strict-ZK global negatives"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
