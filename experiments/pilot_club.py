"""E-CLUB: clustering-of-bandits (CLUB, hard drone clusters) vs continuous low-rank
(RewardCF) vs soft memory-CF (KNNCF) vs tabular/UCB, under persistent masking. Question:
does DISCRETE clustering of agents match CONTINUOUS low-rank factorization for unseen-pair
generalization? Same masked harness (run_masked), guessed d_hat, 8 seeds, rho sweep.
Writes docs/CLUB.md. ZK: every method consumes only the masked, noisy broadcast."""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_c11_masking import run_masked
from pilot_baselines import CLUB
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RHOS = [1.0, 0.5, 0.25]
SEEDS = list(range(8))
RNG = np.random.RandomState(0)

REG = {
    "RewardCF": pc.REGISTRY["RewardCF"],          # continuous low-rank (ours)
    "CLUB":     (CLUB, dict(eps=0.2, sim_thresh=0.4, min_co=3)),  # hard drone clustering
    "KNNCF":    pc.REGISTRY["KNNCF"],             # soft memory-CF
    "Tabular":  pc.REGISTRY["Tabular"],           # no transfer
    "UCBIndep": pc.REGISTRY["UCBIndep"],          # no-structure floor
}
ORDER = ["RewardCF", "CLUB", "KNNCF", "Tabular", "UCBIndep"]


def _job(args):
    nm, rho, seed = args
    Cls, hp = REG[nm]
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    o, u, q = run_masked(Cls, hp, w, pc.T, seed, pc.SO, pc.SB, pc.CAND, pc.D_HAT, rho)
    return nm, rho, seed, float(o), float(u)


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v is not None], float)
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, rho, s) for rho in RHOS for nm in ORDER for s in SEEDS]
    raw = {str(rho): {nm: {"overall": [None] * len(SEEDS), "unseen": [None] * len(SEEDS)}
                      for nm in ORDER} for rho in RHOS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, rho, s, o, u = fut.result()
            raw[str(rho)][nm]["overall"][s] = o
            raw[str(rho)][nm]["unseen"][s] = u
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("club", {
        "meta": {"experiment": "CLUB (hard drone clustering) vs continuous low-rank vs memory-CF vs tabular",
                 "methods": ORDER, "rhos": RHOS, "seeds": SEEDS, "m": pc.M, "n": pc.N, "d": pc.D,
                 "d_hat": pc.D_HAT, "T": pc.T, "cand": pc.CAND,
                 "metric": "overall + unseen-pair skill under persistent masking, per rho"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# CLUB (clustering of bandits) vs continuous low-rank, under masking\n",
         "Does DISCRETE agent-clustering match CONTINUOUS low-rank factorization for unseen-pair "
         "generalization? Same masked harness, guessed d_hat=%d, 8 seeds, bootstrap 95%% CI.\n" % pc.D_HAT]
    for metric, title in [("unseen", "Unseen-pair skill (the categorical claim)"),
                          ("overall", "Overall skill")]:
        L.append("## %s\n" % title)
        L.append("| method | " + " | ".join("rho=%.2f" % r for r in RHOS) + " |")
        L.append("|" + "---|" * (len(RHOS) + 1))
        for nm in ORDER:
            cells = [cell(raw[str(r)][nm][metric]) for r in RHOS]
            lab = "**%s**" % nm if nm == "RewardCF" else nm
            L.append("| %s | %s |" % (lab, " | ".join(cells)))
        L.append("")
    L.append("Read: if CLUB (hard clusters) trails RewardCF (continuous low-rank) on unseen, the "
             "personalization lives in CONTINUOUS factor directions that discrete grouping coarsens "
             "away; if it matches KNNCF, hard-vs-soft grouping is a wash. Both clustering methods should "
             "still beat the structure-free floor (Tabular/UCBIndep ~ 0 on unseen).\n")
    out_md = os.path.join(ROOT, "docs", "CLUB.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
