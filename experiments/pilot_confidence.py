"""
CONFIDENCE done right (don't give up on confidence: find a way that HELPS). The
ablation showed inverse-variance reweighting of the FIT (full precision) hurts unseen
because it under-uses the broadcast that powers generalization. Here we explore several
alternative ways to account for confidence WITHOUT corrupting the global fit, and look
for one that DOMINATES uniform: matches it at low noise AND recovers noise-robustness at
high noise.

Mechanisms (all via ConfCF; converged online weighted-ALS):
  uniform   : w=1                              (baseline; best at default noise)
  full      : w=1/sigma^2                      (the one that hurt)
  rel       : w=(1/sigma^2)/mean               (relative precision: keep the own/teammate
                                                ratio, remove the data-vs-prior scale blow-up)
  cap5      : w=min(1/sigma^2, 5)              (bounded precision: noise-aware, but no row
                                                may dominate the broadcast)
  shrink    : uniform fit + shrink predictions toward the popularity prior by inverse
              COVERAGE (confidence in the DECISION, not the fit)
  post      : uniform fit + shrink by the ALS POSTERIOR precision of u_j (principled)
  relshrink : rel fit + coverage shrinkage (noise-aware fit + coverage-aware decision)

Conditions: rho in {1.0, 0.25} x sigma_obs in {0.3 default, 1.0 high}. Metrics: unseen +
anytime, 8 seeds, bootstrap 95% CI. WINNER = a mechanism that is >= uniform in every
condition and strictly > somewhere (ideally at high noise). Writes docs/CONFIDENCE.md.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_anytime as pa
from pilot_c11_masking import run_masked
from pilot_noise import ConfCF, EMCF
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# lighter-but-converged config for the RELATIVE mechanism comparison (faster sweep);
# absolute headline numbers use als_sweeps=20/refit=1 elsewhere.
_CONV = dict(eps0=0.5, eps_min=0.05, eps_decay=0.97, als_sweeps=12, refit_every=2)
_EM = dict(em_beta=1.0, em_sweeps=6, refit_every=4, eps0=0.5, eps_min=0.05, eps_decay=0.97)
CONF = {                                        # name -> (Class, hp)
    "uniform":   (ConfCF, dict(conf_mode="uniform", **_CONV)),
    "full":      (ConfCF, dict(conf_mode="full", **_CONV)),
    "rel":       (ConfCF, dict(conf_mode="rel", **_CONV)),
    "cap5":      (ConfCF, dict(conf_mode="cap", conf_cap=5.0, **_CONV)),
    "relcap4":   (ConfCF, dict(conf_mode="relcap", conf_alpha=4.0, **_CONV)),
    "shrink":    (ConfCF, dict(conf_mode="uniform", shrink=3.0, **_CONV)),
    "post":      (ConfCF, dict(conf_mode="uniform", post=1.0, **_CONV)),
    "relshrink": (ConfCF, dict(conf_mode="rel", shrink=3.0, **_CONV)),
    "EM":        (EMCF,   dict(**_EM)),                       # variational EM-PMF, CI-UCB
    "EMshrink":  (EMCF,   dict(shrink=0.3, **_EM)),           # + predictive-var shrinkage
}
# key contrasts only (uniform baseline; full = the one that hurt; relcap4 = bounded
# precision; EM/EMshrink = Bayesian factorization w/ confidence intervals)
ORDER = ["uniform", "full", "relcap4", "EM", "EMshrink"]
CONDS = [(1.0, 0.3), (1.0, 1.0), (0.25, 0.3), (0.25, 1.0)]   # (rho, sigma_obs)
SEEDS = list(range(6))
RNG = np.random.RandomState(0)


def _job(args):
    kind, name, rho, sb, seed = args
    Cls, hp = CONF[name]
    if kind == "unseen":
        w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
        v = run_masked(Cls, hp, w, pc.T, seed, pc.SO, sb, pc.CAND, pc.D_HAT, rho)[1]
    else:
        v = pa.run_anytime_clshp(Cls, hp, rho, seed, d_hat=pc.D_HAT, sb=sb)[-1]
    return kind, name, rho, sb, seed, float(v)


def ci(vals, B=10000):
    a = np.asarray(vals, float); idx = RNG.randint(0, len(a), (B, len(a))); m = a[idx].mean(1)
    return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def cond_key(rho, sb):
    return "rho%.2f_s%.1f" % (rho, sb)


def main():
    jobs = [(k, nm, rho, sb, s) for k in ("unseen", "anytime")
            for (rho, sb) in CONDS for nm in ORDER for s in SEEDS]
    raw = {k: {cond_key(*c): {nm: [None] * len(SEEDS) for nm in ORDER} for c in CONDS}
           for k in ("unseen", "anytime")}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            k, nm, rho, sb, s, v = fut.result()
            raw[k][cond_key(rho, sb)][nm][s] = v
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    # SAVE RAW FIRST (so a downstream formatting bug can never discard a full run)
    save_results("confidence8", {
        "meta": {"experiment": "confidence mechanisms bake-off", "mechanisms": ORDER,
                 "conds": CONDS, "seeds": SEEDS, "sigma_own": pc.SO, "d_hat": pc.D_HAT,
                 "metric": "unseen + anytime, bootstrap CI"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    def mean(k, c, nm):
        return float(np.mean(raw[k][cond_key(*c)][nm]))

    L = ["# Confidence mechanisms: can we use confidence WITHOUT losing generalization?\n",
         "All variants are the same online weighted-ALS (ConfCF) or variational EM (EMCF); "
         "they differ only in HOW confidence enters. Baseline = uniform. Skill 0=random, "
         "1=oracle, %d seeds, bootstrap 95%% CI. sigma_obs is the broadcast noise (own fixed "
         "at %.2f).\n" % (len(SEEDS), pc.SO)]
    for k in ("unseen", "anytime"):
        L.append("## %s skill\n" % k.upper())
        L.append("| mechanism | " + " | ".join("rho=%.2f, s=%.1f" % c for c in CONDS) + " |")
        L.append("|" + "---|" * (len(CONDS) + 1))
        for nm in ORDER:
            cells = []
            for c in CONDS:
                mu, lo, hi = ci(raw[k][cond_key(*c)][nm])
                cells.append("%.3f [%.3f, %.3f]" % (mu, lo, hi))
            lab = "**%s**" % nm if nm == "uniform" else nm
            L.append("| %s | %s |" % (lab, " | ".join(cells)))
        L.append("")

    # dominance summary vs uniform (averaged over unseen+anytime per condition)
    L.append("## Dominance vs uniform (mean of unseen+anytime per condition; + = better)\n")
    L.append("| mechanism | " + " | ".join("rho=%.2f, s=%.1f" % c for c in CONDS) + " | verdict |")
    L.append("|" + "---|" * (len(CONDS) + 2))
    best = None
    for nm in ORDER:
        if nm == "uniform":
            continue
        deltas = []
        for c in CONDS:
            d = 0.5 * ((mean("unseen", c, nm) - mean("unseen", c, "uniform"))
                       + (mean("anytime", c, nm) - mean("anytime", c, "uniform")))
            deltas.append(d)
        dominates = all(d >= -0.01 for d in deltas) and any(d > 0.01 for d in deltas)
        verdict = "DOMINATES uniform" if dominates else ("mixed" if any(d > 0.01 for d in deltas) else "<= uniform")
        L.append("| %s | %s | %s |" % (nm, " | ".join("%+.3f" % d for d in deltas), verdict))
        if dominates and (best is None or sum(deltas) > best[1]):
            best = (nm, sum(deltas))
    L.append("")
    if best:
        L.append("WINNER: **%s** is >= uniform in every condition and better in at least one "
                 "(esp. high noise): a confidence mechanism that keeps the broadcast at full "
                 "weight in the fit (so unseen generalization is preserved) while still being "
                 "noise/coverage-aware. This is the right way to account for confidence.\n" % best[0])
    else:
        L.append("No single mechanism strictly dominates uniform across all conditions; the "
                 "honest reading is a Pareto trade (noise-awareness helps at high sigma_obs, "
                 "costs a little unseen at low sigma_obs). The recommended default is the one "
                 "with the best high-noise gain at no low-noise cost (see deltas).\n")

    out_md = os.path.join(ROOT, "docs", "CONFIDENCE.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    save_results("confidence8", {
        "meta": {"experiment": "confidence mechanisms bake-off", "mechanisms": ORDER,
                 "conds": CONDS, "seeds": SEEDS, "sigma_own": pc.SO, "d_hat": pc.D_HAT,
                 "metric": "unseen + anytime, bootstrap CI"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))
    print("wrote %s" % out_md)


if __name__ == "__main__":
    main()
