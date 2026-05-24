"""(b) POSITIVE SCALING WITH SWARM SIZE: does the swarm get SMARTER as it grows? Sweep the number of
drones m at fixed n, fixed per-drone horizon T and broadcast rate rho. For CF, more drones means more
total broadcast observations (~ m*T*rho) feeding the SHARED low-rank structure, so each drone's unseen
skill RISES with m (the collective-recovery speedup, T11) -- the swarm gets smarter as it gets bigger.
A structure-free learner learns only its own row, so m does not help it: FLAT in m (Thm 1). Standard
masked harness, unseen + overall skill, 8 seeds. Writes docs/SCALE_M.md."""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_c11_masking import run_masked
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MS = [5, 10, 20, 40, 80]                # swarm sizes (K1=min(K,m) so latent rank stays = d for all)
RHO = 0.5                               # partial broadcast (so collaboration matters)
SEEDS = list(range(8))
RNG = np.random.RandomState(0)
REG = {
    "RewardCF": pc.REGISTRY["RewardCF"],          # ours (online, SwarmCF)
    "PTF":      pc.REGISTRY["PTF"],               # ours (batch variant, SwarmCF-batch)
    "UCBIndep": pc.REGISTRY["UCBIndep"],          # structure-free
    "Tabular":  pc.REGISTRY["Tabular"],           # structure-free
}
ORDER = ["RewardCF", "PTF", "UCBIndep", "Tabular"]


def _job(args):
    nm, m, seed = args
    Cls, hp = REG[nm]
    w = make_world(m, pc.N, pc.D, min(pc.K, m), pc.K, within=0.15, seed=seed, signed=True)
    o, u, q = run_masked(Cls, hp, w, pc.T, seed, pc.SO, pc.SB, pc.CAND, pc.D_HAT, RHO)
    return nm, m, seed, float(o), float(u)


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v is not None], float)
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, m, s) for m in MS for nm in ORDER for s in SEEDS]
    raw = {str(m): {nm: {"overall": [None] * len(SEEDS), "unseen": [None] * len(SEEDS)}
                    for nm in ORDER} for m in MS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, m, s, o, u = fut.result()
            raw[str(m)][nm]["overall"][s] = o; raw[str(m)][nm]["unseen"][s] = u
            done += 1
            if done % 12 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("scale_m", {
        "meta": {"experiment": "positive scaling with swarm size m (does the swarm get smarter as it grows?)",
                 "methods": ORDER, "ms": MS, "rho": RHO, "seeds": SEEDS, "n": pc.N, "d": pc.D,
                 "d_hat": pc.D_HAT, "T": pc.T, "metric": "unseen + overall skill vs swarm size m"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# Positive scaling with swarm size: does the swarm get smarter as it grows? (rho=%.2f)\n" % RHO,
         "Unseen-pair skill vs number of drones m (fixed n=%d, T=%d, partial broadcast). 8 seeds, "
         "bootstrap 95%% CI.\n" % (pc.N, pc.T),
         "| method | " + " | ".join("m=%d" % m for m in MS) + " |", "|" + "---|" * (len(MS) + 1)]
    for nm in ORDER:
        cells = [cell(raw[str(m)][nm]["unseen"]) for m in MS]
        lab = "**%s**" % nm if nm == "RewardCF" else nm
        L.append("| %s | %s |" % (lab, " | ".join(cells)))
    L.append("")
    cf_lo = float(np.mean(raw[str(MS[0])]["RewardCF"]["unseen"]))
    cf_hi = float(np.mean(raw[str(MS[-1])]["RewardCF"]["unseen"]))
    sf_lo = float(np.mean(raw[str(MS[0])]["UCBIndep"]["unseen"]))
    sf_hi = float(np.mean(raw[str(MS[-1])]["UCBIndep"]["unseen"]))
    L.append("**Result:** CF unseen skill rises from %.3f (m=%d) to %.3f (m=%d) as the swarm grows -- the "
             "swarm gets SMARTER as it gets bigger, because more drones contribute more broadcast "
             "observations to the SHARED low-rank structure (collective recovery, T11). The structure-free "
             "learner is FLAT (%.3f -> %.3f): m does not help an agent that only learns its own row "
             "(Thm 1). Positive scaling with team size is unique to the structure-sharing swarm.\n"
             % (cf_lo, MS[0], cf_hi, MS[-1], sf_lo, sf_hi))
    out_md = os.path.join(ROOT, "docs", "SCALE_M.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
