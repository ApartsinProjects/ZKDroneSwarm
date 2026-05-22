"""
P1-4: 20-seed headline confirmation with bootstrap 95% CIs, for the camera-ready
table. Methods: UCBIndep (no-structure), PTF (strongest competitor), and our
RewardCF / HybridCFconv / ActiveCFconv. Metrics: unseen-pair and anytime skill at
rho in {1.0, 0.25}. Writes docs/HEADLINE_TABLE.md.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_anytime as pa
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["UCBIndep", "PTF", "RewardCF", "HybridCFconv", "ActiveCFconv"]
RHOS = [1.0, 0.25]
SEEDS = list(range(20))
RNG = np.random.RandomState(0)


def _job(args):
    kind, nm, rho, s = args
    if kind == "unseen":
        return kind, nm, rho, s, pc._run_cell((nm, rho, s))[4]
    return kind, nm, rho, s, float(np.asarray(pa.run_anytime((nm, rho, s))[3])[-1])


def ci(vals, B=10000):
    a = np.asarray(vals); idx = RNG.randint(0, len(a), (B, len(a))); m = a[idx].mean(1)
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
            if done % 40 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))
    L = ["# Headline results (20 seeds, bootstrap 95% CI)\n",
         "skill = (method - random)/(oracle - random); 0 = random, 1 = oracle. "
         "rho = fraction of broadcast observed. Ours in bold.\n"]
    for k in ("unseen", "anytime"):
        L.append("## %s skill" % ("UNSEEN-pair" if k == "unseen" else "ANYTIME"))
        L.append("| method | rho=1.0 | rho=0.25 |"); L.append("|---|---|---|")
        for nm in METHODS:
            cells = []
            for rho in RHOS:
                mu, lo, hi = ci(raw[k][str(rho)][nm]); cells.append("%.3f [%.3f, %.3f]" % (mu, lo, hi))
            nmd = "**%s**" % nm if ("CF" in nm) else nm
            L.append("| %s | %s | %s |" % (nmd, cells[0], cells[1]))
        L.append("")
    out_md = os.path.join(ROOT, "docs", "HEADLINE_TABLE.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    save_results("headline20", {"meta": {"experiment": "20-seed headline CIs", "methods": METHODS,
                 "rhos": RHOS, "seeds": SEEDS, "metric": "unseen + anytime, bootstrap CI"},
                 "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))
    print("wrote %s" % out_md)


if __name__ == "__main__":
    main()
