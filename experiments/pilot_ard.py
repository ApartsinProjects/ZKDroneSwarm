"""
ARD rank self-determination (backlog C7 / the rank-adaptation question). Does Automatic
Relevance Determination let the method find the rank from data, removing the guessed-rank
hyperparameter? We set d_hat ABSURDLY high (20, true d=5) and check that ARD-EMCF (a) keeps
skill (does not overfit the extra dimensions) and (b) recovers an effective rank ~5, while
plain EMCF at d_hat=20 over-parametrizes. Baseline: RewardCF at the tuned d_hat=8. rho=1.0,
8 seeds, bootstrap 95% CI. Writes docs/ARD.md. Saves raw before formatting.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_anytime as pa
from pilot_noise import RewardCF, EMCF
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_EM = dict(em_beta=1.0, em_sweeps=6, refit_every=4, eps0=0.5, eps_min=0.05, eps_decay=0.97)
_CONV = dict(eps0=0.5, eps_min=0.05, eps_decay=0.97, als_sweeps=12, refit_every=2)
VARIANTS = [   # (label, Cls, hp, d_hat)
    ("RewardCF (d_hat=8)",   RewardCF, dict(**_CONV),            8),
    ("EMCF (d_hat=8)",       EMCF,     dict(**_EM),              8),
    ("EMCF (d_hat=20)",      EMCF,     dict(**_EM),             20),
    ("ARD-EMCF (d_hat=8)",   EMCF,     dict(ard=True, **_EM),    8),
    ("ARD-EMCF (d_hat=20)",  EMCF,     dict(ard=True, **_EM),   20),
]
RHO = 1.0
SEEDS = list(range(8))
RNG = np.random.RandomState(0)


def train_unseen_effrank(Cls, hp, d_hat, seed):
    """Train m learners under masking; return (unseen-pair skill, mean effective rank)."""
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m, n = R.shape
    T, cand, so, sb = pc.T, pc.CAND, pc.SO, pc.SB
    rng = np.random.RandomState(seed + 999)
    M = rng.rand(m, m) < RHO; np.fill_diagonal(M, True)
    learners = [Cls(m, n, d_hat, i, seed + 7 * i + 1, **hp) for i in range(m)]
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
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
            if len(unseen) >= cand:
                off = g.choice(unseen, size=cand, replace=False)
                ung.append(R[k, off[int(np.argmax(preds[k][off]))]])
                uno.append(R[k, off].max()); unr.append(R[k, off].mean())
    gm, om, rm = np.mean(ung), np.mean(uno), np.mean(unr)
    unseen_skill = (gm - rm) / max(om - rm, 1e-6)
    er = float(np.mean([ln.eff_rank() for ln in learners if hasattr(ln, "eff_rank")])) \
        if hasattr(learners[0], "eff_rank") else float("nan")
    return float(unseen_skill), er


def _job(args):
    kind, vi, seed = args
    label, Cls, hp, d_hat = VARIANTS[vi]
    if kind == "unseen":
        u, er = train_unseen_effrank(Cls, hp, d_hat, seed)
        return kind, vi, seed, u, er
    a = pa.run_anytime_clshp(Cls, hp, RHO, seed, d_hat=d_hat)[-1]
    return kind, vi, seed, float(a), float("nan")


def ci(vals, B=10000):
    a = np.asarray(vals, float); idx = RNG.randint(0, len(a), (B, len(a))); m = a[idx].mean(1)
    return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(k, vi, s) for k in ("unseen", "anytime") for vi in range(len(VARIANTS)) for s in SEEDS]
    raw = {k: {vi: [None] * len(SEEDS) for vi in range(len(VARIANTS))} for k in ("unseen", "anytime")}
    er = {vi: [None] * len(SEEDS) for vi in range(len(VARIANTS))}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            kind, vi, seed, v, e = fut.result()
            raw[kind][vi][seed] = v
            if kind == "unseen" and np.isfinite(e):
                er[vi][seed] = e
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("ard8", {"meta": {"experiment": "ARD rank self-determination",
                 "variants": [v[0] for v in VARIANTS], "true_d": pc.D, "rho": RHO,
                 "seeds": SEEDS, "metric": "unseen+anytime + recovered eff rank"},
                 "raw": raw, "eff_rank": er}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# ARD: does the method find the rank itself? (true d=%d)\n" % pc.D,
         "Set the guessed rank d_hat absurdly high (20) and check ARD-EMCF keeps skill AND "
         "recovers an effective rank ~%d, removing the rank hyperparameter. rho=1.0, 8 seeds, "
         "bootstrap 95%% CI.\n" % pc.D,
         "| variant | unseen skill | anytime skill | recovered eff. rank |",
         "|---|---|---|---|"]
    for vi, (label, _, _, _) in enumerate(VARIANTS):
        ers = [e for e in er[vi] if e is not None]
        ercell = "%.1f" % np.mean(ers) if ers else "-- (no ARD)"
        L.append("| %s | %s | %s | %s |" % (label, cell(raw["unseen"][vi]),
                                            cell(raw["anytime"][vi]), ercell))
    L.append("")
    L.append("Read: if ARD works, ARD-EMCF (d_hat=20) should (a) match ARD-EMCF/EMCF at the tuned "
             "d_hat=8 and the RewardCF baseline on skill (no overfit from the 15 extra dimensions), "
             "and (b) report a recovered effective rank near the true d=%d, so you never have to know "
             "or tune the rank.\n" % pc.D)

    out_md = os.path.join(ROOT, "docs", "ARD.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
