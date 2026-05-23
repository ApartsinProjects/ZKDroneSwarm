"""Does the 'noise level known' assumption matter? Compare three RewardCF variants under homogeneous
vs heterogeneous teammate noise (rho=1.0): (1) KNOWN-sigma precision (given rvar=sigma^2, w=1/sigma^2);
(2) UNIFORM (ignores sigma); (3) EST-sigma (RewardCFEstSigma: estimates per-source sigma^2 from
prediction residuals, empirical Bayes). If EST-sigma MATCHES known-sigma, the assumption is removable
(sigma can be estimated, not assumed). Reuses the precision-hetero harness. Writes docs/EST_SIGMA.md."""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_precision_hetero as pph
from pilot_noise import RewardCF, RewardCFEstSigma
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALS = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=10, refit_every=3)
REG = {
    "known-sigma": (RewardCF, dict(precision=True, **_ALS)),       # given rvar=sigma^2 (assumes known)
    "uniform": (RewardCF, dict(precision=False, **_ALS)),          # ignores sigma
    "est-sigma": (RewardCFEstSigma, dict(**_ALS)),                 # estimates sigma from residuals
}
ORDER = ["known-sigma", "uniform", "est-sigma"]
SCEN = ["homog", "hetero"]
SEEDS = list(range(8))
RNG = np.random.RandomState(0)


def _job(args):
    nm, scen, seed = args
    Cls, hp = REG[nm]
    u, a = pph._run(Cls, hp, scen, seed)
    return nm, scen, seed, float(u), float(a)


def ci(vals, B=10000):
    arr = np.asarray([v for v in vals if v is not None], float)
    idx = RNG.randint(0, len(arr), (B, len(arr))); mb = arr[idx].mean(1)
    return arr.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, sc, s) for sc in SCEN for nm in ORDER for s in SEEDS]
    raw = {sc: {nm: {"unseen": [None] * len(SEEDS), "anytime": [None] * len(SEEDS)} for nm in ORDER} for sc in SCEN}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, sc, s, u, a = fut.result()
            raw[sc][nm]["unseen"][s] = u
            raw[sc][nm]["anytime"][s] = a
            done += 1
            if done % 12 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("estsigma", {
        "meta": {"experiment": "known vs uniform vs estimated sigma (is the noise-known assumption load-bearing?)",
                 "methods": ORDER, "scenarios": SCEN, "seeds": SEEDS, "m": pc.M, "n": pc.N, "d": pc.D,
                 "d_hat": pc.D_HAT, "metric": "unseen + anytime skill, homog vs hetero teammate noise"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# Is the 'noise level known' assumption load-bearing? (est-sigma vs known-sigma vs uniform)\n",
         "RewardCF with KNOWN sigma (given rvar) vs UNIFORM (ignores sigma) vs EST-sigma (estimates "
         "per-source sigma from residuals). rho=1.0; homog (all sigma=1.0) vs hetero (half 0.1, half 1.9). "
         "8 seeds, bootstrap 95%% CI.\n"]
    for metric in ["unseen", "anytime"]:
        L.append("## %s skill\n" % metric)
        L.append("| method | homog | hetero |")
        L.append("|---|---|---|")
        for nm in ORDER:
            L.append("| %s | %s | %s |" % (nm, cell(raw["homog"][nm][metric]), cell(raw["hetero"][nm][metric])))
        L.append("")
    du = float(np.mean(raw["hetero"]["est-sigma"]["unseen"]) - np.mean(raw["hetero"]["known-sigma"]["unseen"]))
    L.append("est-sigma minus known-sigma (hetero, unseen): %+.3f\n" % du)
    L.append("Read: if EST-sigma MATCHES known-sigma (small gap) and both >= uniform under HETERO noise, "
             "the 'noise level known' assumption is NOT load-bearing, sigma can be estimated from "
             "residuals; and since uniform is competitive, the categorical headline needs no sigma at all.\n")
    out_md = os.path.join(ROOT, "docs", "EST_SIGMA.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
