"""
Precision weighting under HETEROGENEOUS teammate noise (SANITY check for the
'uniform beats precision' negative; H11(b)).

Background: the ablation (8.12) found that inverse-variance (precision, w=1/sigma^2) fit
weighting does NOT beat UNIFORM at the default HOMOGENEOUS broadcast noise; coverage, not
noise, is the binding constraint there, and 'full' precision over-trusts a drone's own
clean rewards and under-uses the broadcast that powers generalization. That is an honest
but easily-misread negative. Obvious-expected SANITY: if teammates DIFFER in observation
noise (some broadcasts clean, some garbage), precision MUST help, it down-weights the
garbage. If uniform still won there, the precision implementation would be suspect.

Controlled design (matched MEAN broadcast noise, vary only HETEROGENEITY):
  homog  : every teammate sigma_obs = 1.0
  hetero : half the teammates sigma_obs = 0.1 (clean), half = 1.9 (noisy)  (mean ~1.0)
rho=1.0 (full broadcast, isolate noise from masking). Own noise sigma_own fixed.
Methods (ConfCF fit-weight modes): uniform (w=1) / full (w=1/sigma^2) / relcap
(ratio-bounded precision). Metric: unseen-pair + anytime skill, 8 seeds, bootstrap 95% CI.
Expect: precision's edge over uniform is ~0 (or negative) in HOMOG and clearly POSITIVE in
HETERO. Writes docs/PRECISION_HETERO.md. Saves raw first.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_noise import ConfCF
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CONV = dict(eps0=0.5, eps_min=0.05, eps_decay=0.97, als_sweeps=12, refit_every=2)
REG = {
    "uniform": (ConfCF, dict(conf_mode="uniform", **_CONV)),
    "full":    (ConfCF, dict(conf_mode="full", **_CONV)),
    "relcap":  (ConfCF, dict(conf_mode="relcap", conf_alpha=8.0, **_CONV)),
}
ORDER = ["uniform", "full", "relcap"]
SCEN = ["homog", "hetero"]
SEEDS = list(range(8))
RNG = np.random.RandomState(0)
SIG_MEAN = 1.0                     # matched mean broadcast noise across scenarios


def _sigmas(scenario, m, rng):
    s = np.full(m, SIG_MEAN, float)
    if scenario == "hetero":
        half = m // 2
        s[:half] = 0.1; s[half:] = 1.9       # clean / noisy split, same mean ~1.0
        rng.shuffle(s)
    return s


def _run(Cls, hp, scenario, seed):
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m, n = R.shape
    T, cand, so = pc.T, pc.CAND, pc.SO
    rng = np.random.RandomState(seed + 999)
    sig = _sigmas(scenario, m, rng)              # per-teammate broadcast noise
    learners = [Cls(m, n, pc.D_HAT, i, seed + 7 * i + 1, **hp) for i in range(m)]
    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        real[t] = float(true_r.sum())
        orac[t] = float(sum(R[i, cand_sets[i]].max() for i in range(m)))
        rnd[t] = float(sum(R[i, cand_sets[i]].mean() for i in range(m)))
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):                    # rho=1.0: full broadcast, per-teammate noise sig[k]
                if k != i:
                    revealed[k] = true_r[k] + rng.normal(0, sig[k]); rvar[k] = sig[k] ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)

    cr, co, cd = real.cumsum(), orac.cumsum(), rnd.cumsum()
    anytime = float((cr[-1] - cd[-1]) / max(co[-1] - cd[-1], 1e-9))

    g = np.random.RandomState(seed + 555)
    ung, uno, unr = [], [], []
    for k in range(m):
        preds = learners[k].predict_scores()
        unseen = np.where(~learners[k].pulled_mask())[0]
        if len(unseen) < cand:
            continue
        for _ in range(120):
            off = g.choice(unseen, size=cand, replace=False)
            ung.append(R[k, off[int(np.argmax(preds[off]))]])
            uno.append(R[k, off].max()); unr.append(R[k, off].mean())
    unseen = float((np.mean(ung) - np.mean(unr)) / max(np.mean(uno) - np.mean(unr), 1e-6))
    return unseen, anytime


def _job(args):
    name, scen, seed = args
    Cls, hp = REG[name]
    u, a = _run(Cls, hp, scen, seed)
    return name, scen, seed, u, a


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v == v], float)
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals)
    return "n/a" if mu != mu else "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(nm, sc, s) for sc in SCEN for nm in ORDER for s in SEEDS]
    raw = {sc: {nm: {"unseen": [None] * len(SEEDS), "anytime": [None] * len(SEEDS)}
                for nm in ORDER} for sc in SCEN}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, sc, s, u, a = fut.result()
            raw[sc][nm]["unseen"][s] = u; raw[sc][nm]["anytime"][s] = a
            done += 1
            if done % 12 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("precisionhetero8", {
        "meta": {"experiment": "precision weighting under heterogeneous teammate noise (sanity)",
                 "methods": ORDER, "scenarios": SCEN, "seeds": SEEDS, "sig_mean": SIG_MEAN,
                 "m": pc.M, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT, "T": pc.T, "sigma_own": pc.SO,
                 "metric": "unseen + anytime, full broadcast, per-teammate noise"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# Precision weighting under heterogeneous teammate noise (sanity check)\n",
         "Matched MEAN broadcast noise; only the HETEROGENEITY varies. homog = every teammate "
         "sigma_obs=1.0; hetero = half sigma_obs=0.1 (clean), half=1.9 (noisy). rho=1.0 (full "
         "broadcast). Does precision (1/sigma^2) fit weighting beat uniform when sources DIFFER in "
         "noise? 8 seeds, bootstrap 95%% CI.\n"]
    for metric, title in (("unseen", "Unseen-pair skill"), ("anytime", "Anytime earned skill")):
        L.append("## %s\n" % title)
        L.append("| method | " + " | ".join(SCEN) + " |")
        L.append("|" + "---|" * (len(SCEN) + 1))
        for nm in ORDER:
            cells = [cell(raw[sc][nm][metric]) for sc in SCEN]
            L.append("| %s | %s |" % (nm, " | ".join(cells)))
        # precision-minus-uniform deltas (the sanity signal)
        for prec in ("full", "relcap"):
            d = ["%+.3f" % (np.mean([v for v in raw[sc][prec][metric] if v is not None])
                           - np.mean([v for v in raw[sc]["uniform"][metric] if v is not None]))
                 for sc in SCEN]
            L.append("| (%s - uniform) | %s |" % (prec, " | ".join(d)))
        L.append("")
    L.append("Sanity: precision's edge over uniform should be ~0 or negative in HOMOG (the known "
             "result: coverage, not noise, binds; 'full' over-trusts own clean data) and clearly "
             "POSITIVE in HETERO (it correctly down-weights the noisy half). A positive HETERO delta "
             "confirms the precision machinery works and SCOPES the negative: precision is the right "
             "call exactly when observation sources differ in reliability.\n")

    out_md = os.path.join(ROOT, "docs", "PRECISION_HETERO.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
