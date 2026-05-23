"""
EMCF predictive-interval CALIBRATION (H4 / sanity H11d): is the model's confidence REAL?

EMCF (variational-Bayes PMF) predicts each unseen reward with a MEAN (muU . muP_i) and a
predictive VARIANCE (Var = mi^T S_j mi + mu_j^T S_i mu_j + tr(S_i S_j)). We USE that
variance to drive UCB exploration and shrinkage, so it must be more than decorative. Two
obvious-expected sanity properties:

  DISCRIMINATION (the property that matters for UCB/shrinkage): pairs the model says it is
    LESS sure about must actually have LARGER error. Bin unseen pairs by predicted sd into
    quintiles; the actual RMSE must RISE across bins. (A constant/meaningless sd gives a
    flat curve.)
  CALIBRATION (coverage): a nominal X% predictive interval should contain the truth about
    X% of the time. Mean-field VI is known to be somewhat OVER-confident, so we report the
    empirical coverage honestly rather than assume it.

We also contrast EMCF's posterior sd against a NAIVE homoskedastic sd (one global residual
std for every pair): the naive one has zero discrimination by construction, so EMCF's
discrimination curve must beat it for the confidence to be worth anything.

rho=1.0 (full broadcast), standard world, EMCF with shrink=0 (raw posterior interval),
8 seeds, bootstrap 95% CI. Writes docs/CALIBRATION.md. Saves raw first.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_noise import EMCF
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RHO = 1.0
SEEDS = list(range(8))
_HP = dict(lam=1.0, em_sweeps=8, refit_every=2, em_beta=1.0, shrink=0.0,
           eps0=0.5, eps_min=0.05, eps_decay=0.97)
NOMINAL = [0.50, 0.80, 0.90, 0.95]
ZQ = {0.50: 0.674, 0.80: 1.282, 0.90: 1.645, 0.95: 1.960}
NBIN = 5
RNG = np.random.RandomState(0)


def _train_collect(seed):
    """Train EMCF for one episode; return unseen-pair (prediction, posterior sd, true reward)."""
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m, n = R.shape
    T, cand, so, sb = pc.T, pc.CAND, pc.SO, pc.SB
    rng = np.random.RandomState(seed + 999)
    Mask = rng.rand(m, m) < RHO; np.fill_diagonal(Mask, True)
    learners = [EMCF(m, n, pc.D_HAT, i, seed + 7 * i + 1, **_HP) for i in range(m)]
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i and Mask[i, k]:
                    revealed[k] = true_r[k] + rng.normal(0, sb); rvar[k] = sb ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    preds, sds, trues = [], [], []
    for i in range(m):
        mu = learners[i].predict_scores()             # shrink=0 -> raw posterior mean muU.muP_i
        sd = np.sqrt(learners[i]._predvar())
        uns = np.where(~learners[i].pulled_mask())[0]
        if len(uns) == 0:
            continue
        preds.append(mu[uns]); sds.append(sd[uns]); trues.append(R[i, uns])
    return np.concatenate(preds), np.concatenate(sds), np.concatenate(trues)


def _job(seed):
    pred, sd, true = _train_collect(seed)
    err = np.abs(true - pred)
    z = (true - pred) / np.maximum(sd, 1e-9)
    cov = [float(np.mean(np.abs(z) < ZQ[lvl])) for lvl in NOMINAL]      # empirical coverage
    # DISCRIMINATION: quintiles of predicted sd -> actual RMSE per bin (EMCF vs naive constant sd)
    order = np.argsort(sd); bins = np.array_split(order, NBIN)
    binsd = [float(np.mean(sd[b])) for b in bins]
    binerr = [float(np.sqrt(np.mean(err[b] ** 2))) for b in bins]
    # naive homoskedastic sd = global RMSE; bin by THAT (random wrt error) -> flat by construction
    naive_sd = float(np.sqrt(np.mean(err ** 2)))
    naive_z = (true - pred) / max(naive_sd, 1e-9)
    naive_cov = [float(np.mean(np.abs(naive_z) < ZQ[lvl])) for lvl in NOMINAL]
    return seed, cov, naive_cov, binsd, binerr


def ci(vals, B=10000):
    a = np.asarray(vals, float)
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def main():
    cov = []; naive_cov = []; binsd = []; binerr = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_job, s) for s in SEEDS]
        for fut in as_completed(futs):
            _, c, nc, bs, be = fut.result()
            cov.append(c); naive_cov.append(nc); binsd.append(bs); binerr.append(be)
    cov = np.array(cov); naive_cov = np.array(naive_cov)
    binsd = np.array(binsd); binerr = np.array(binerr)

    save_results("calibration8", {
        "meta": {"experiment": "EMCF predictive-interval calibration + discrimination",
                 "rho": RHO, "seeds": SEEDS, "nominal": NOMINAL, "nbin": NBIN,
                 "m": pc.M, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT, "T": pc.T},
        "raw": {"coverage": cov.tolist(), "naive_coverage": naive_cov.tolist(),
                "bin_sd": binsd.tolist(), "bin_rmse": binerr.tolist()}},
        results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# EMCF predictive-interval calibration (is the confidence real?)\n",
         "EMCF posterior predictive intervals on UNSEEN pairs, rho=%.1f, 8 seeds, bootstrap 95%% CI.\n" % RHO]
    L.append("## Discrimination: does predicted uncertainty track actual error?\n")
    L.append("Unseen pairs binned into quintiles by EMCF's predicted sd (Q1=most confident -> "
             "Q5=least). A real confidence makes actual RMSE RISE across bins.\n")
    L.append("| quintile | mean predicted sd | actual RMSE |")
    L.append("|---|---|---|")
    for q in range(NBIN):
        sd_m = binsd[:, q].mean(); err_m, err_lo, err_hi = ci(binerr[:, q])
        L.append("| Q%d | %.3f | %.3f [%.3f, %.3f] |" % (q + 1, sd_m, err_m, err_lo, err_hi))
    rise = binerr[:, -1].mean() - binerr[:, 0].mean()
    L.append("\nActual RMSE rises %+.3f from the most-confident to least-confident quintile "
             "(positive => predicted sd is informative, not decorative).\n" % rise)
    L.append("## Calibration: empirical coverage of nominal intervals\n")
    L.append("| nominal | EMCF empirical | naive (constant sd) |")
    L.append("|---|---|---|")
    for q, lvl in enumerate(NOMINAL):
        em_m, em_lo, em_hi = ci(cov[:, q]); nv_m = naive_cov[:, q].mean()
        L.append("| %d%% | %.3f [%.3f, %.3f] | %.3f |" % (int(100 * lvl), em_m, em_lo, em_hi, nv_m))
    L.append("\nRead: a perfectly calibrated model matches nominal exactly; mean-field VI is "
             "typically OVER-confident (empirical < nominal). What matters for UCB/shrinkage is "
             "DISCRIMINATION (above), which does not require perfect coverage, only that more-uncertain "
             "predictions are genuinely worse. The naive constant-sd column has identical 'intervals' "
             "for every pair, so it cannot discriminate at all.\n")

    out_md = os.path.join(ROOT, "docs", "CALIBRATION.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
