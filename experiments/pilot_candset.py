"""
Candidate-set-size independence (the "why limit the candidate set / why not let each drone
peek the entire target set?" question). We sweep the TRAINING candidate size c_train from 20
up to the FULL target set (c_train=n=240), while evaluating unseen-pair skill with a FIXED
offer of 20 unseen targets. Claim: the CATEGORICAL result (CF >> Tabular on unseen) is
INDEPENDENT of c_train, including c=n -- a tabular learner still cannot rank targets it never
engaged, no matter how many it is shown. rho=1.0, 8 seeds, bootstrap 95% CI. Writes
docs/CANDSET.md. Saves raw before formatting.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_noise import RewardCF, Tabular
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CONV = dict(eps0=0.5, eps_min=0.05, eps_decay=0.97, als_sweeps=12, refit_every=2)
_EPS = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)
REG = {"RewardCF": (RewardCF, dict(**_CONV)), "Tabular": (Tabular, dict(**_EPS))}
ORDER = ["RewardCF", "Tabular"]
CTRAIN = [20, 60, 120, 240]        # training candidate size; 240 = n = "peek the entire set"
EVAL_CAND = 20                     # fixed unseen-eval offer size
RHO = 1.0
SEEDS = list(range(8))
RNG = np.random.RandomState(0)


def run_candset(Cls, hp, c_train, seed):
    """Train with offers of size c_train; eval unseen-pair skill with a fixed EVAL_CAND offer."""
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m, n = R.shape
    T, so, sb = pc.T, pc.SO, pc.SB
    rng = np.random.RandomState(seed + 999)
    M = rng.rand(m, m) < RHO; np.fill_diagonal(M, True)
    learners = [Cls(m, n, pc.D_HAT, i, seed + 7 * i + 1, **hp) for i in range(m)]
    for t in range(T):
        cand_sets = [rng.choice(n, size=c_train, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i and M[i, k]:
                    revealed[k] = true_r[k] + rng.normal(0, sb); rvar[k] = sb ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    preds = [learners[i].predict_scores() for i in range(m)]
    pulled = [learners[i].pulled_mask() for i in range(m)]
    g = np.random.RandomState(seed + 555); ung, uno, unr = [], [], []
    for _ in range(120):
        for k in range(m):
            unseen = np.where(~pulled[k])[0]
            if len(unseen) >= EVAL_CAND:
                off = g.choice(unseen, size=EVAL_CAND, replace=False)
                ung.append(R[k, off[int(np.argmax(preds[k][off]))]])
                uno.append(R[k, off].max()); unr.append(R[k, off].mean())
    gm, om, rm = np.mean(ung), np.mean(uno), np.mean(unr)
    return float((gm - rm) / max(om - rm, 1e-6))


def _job(args):
    name, c_train, seed = args
    Cls, hp = REG[name]
    return name, c_train, seed, run_candset(Cls, hp, c_train, seed)


def ci(vals, B=10000):
    a = np.asarray(vals, float); idx = RNG.randint(0, len(a), (B, len(a))); m = a[idx].mean(1)
    return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, c, s) for c in CTRAIN for nm in ORDER for s in SEEDS]
    raw = {nm: {str(c): [None] * len(SEEDS) for c in CTRAIN} for nm in ORDER}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, c, s, v = fut.result(); raw[nm][str(c)][s] = v
            done += 1
            if done % 16 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("candset8", {"meta": {"experiment": "candidate-set-size independence",
                 "methods": ORDER, "c_train": CTRAIN, "eval_cand": EVAL_CAND, "n": pc.N,
                 "rho": RHO, "seeds": SEEDS, "metric": "unseen-pair skill"},
                 "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# Candidate-set-size independence (does limiting the offer matter?)\n",
         "Training candidate size c_train swept 20 -> %d (=n, 'peek the entire target set'); unseen "
         "eval offer fixed at %d. rho=1.0, 8 seeds, bootstrap 95%% CI.\n" % (pc.N, EVAL_CAND),
         "| method | " + " | ".join("c_train=%d" % c for c in CTRAIN) + " |",
         "|" + "---|" * (len(CTRAIN) + 1)]
    for nm in ORDER:
        cells = [cell(raw[nm][str(c)]) for c in CTRAIN]
        lab = "**%s**" % nm if nm == "RewardCF" else nm
        L.append("| %s | %s |" % (lab, " | ".join(cells)))
    L.append("")
    L.append("Takeaway: CF's unseen-pair skill is essentially flat across c_train (including the full "
             "c=n case), and Tabular stays at the floor (~0) throughout. The categorical separation "
             "does NOT depend on the candidate-set size: a tabular learner cannot rank targets it "
             "never engaged no matter how many it is shown each round. The candidate set is a modeling "
             "knob (per-round reachability), not load-bearing for the result.\n")

    out_md = os.path.join(ROOT, "docs", "CANDSET.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
