"""
ChoiceEM test (Section 5.6): does jointly estimating per-teammate choice-informativeness
beat the fixed competence ramp? We sweep the REWARD noise sigma_obs (the choice channel
matters more as rewards get noisier, since model-based CHOICES stay clean) and compare:
  RewardCF  -- reward channel only (no choices)
  ChoiceCF  -- own reward + teammate choices, fixed temporal competence weight
  ChoiceEM  -- own reward + teammate choices, EM-learned per-choice responsibility
Hypothesis: ChoiceEM >= ChoiceCF (better choice weighting), and the choice methods beat
RewardCF at HIGH sigma_obs where rewards are too noisy but choices are not. rho=1.0 (full
broadcast). Metrics: unseen + anytime, 8 seeds, bootstrap 95% CI. Writes docs/CHOICEEM.md.
Saves raw BEFORE formatting.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_anytime as pa
from pilot_c11_masking import run_masked
from pilot_noise import RewardCF, ChoiceCF, ChoiceEM
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CONV = dict(eps0=0.5, eps_min=0.05, eps_decay=0.97, als_sweeps=12, refit_every=2)
REG = {
    "RewardCF": (RewardCF, dict(**_CONV)),
    "ChoiceCF": (ChoiceCF, dict(comp=True, s2c=0.2, n_neg=1, within=True,
                               warm_frac=0.3, T_total=pc.T, **_CONV)),
    "ChoiceEM": (ChoiceEM, dict(tau=0.3, s2c=0.2, n_neg=1, within=True,
                               T_total=pc.T, **_CONV)),
}
ORDER = ["RewardCF", "ChoiceCF", "ChoiceEM"]
SIGMAS = [0.3, 0.6, 1.0]          # broadcast reward noise (own fixed at sigma_own=0.10)
RHO = 1.0
SEEDS = list(range(8))
RNG = np.random.RandomState(0)


def _job(args):
    kind, name, sb, seed = args
    Cls, hp = REG[name]
    if kind == "unseen":
        w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
        v = run_masked(Cls, hp, w, pc.T, seed, pc.SO, sb, pc.CAND, pc.D_HAT, RHO)[1]
    else:
        v = pa.run_anytime_clshp(Cls, hp, RHO, seed, d_hat=pc.D_HAT, sb=sb)[-1]
    return kind, name, sb, seed, float(v)


def ci(vals, B=10000):
    a = np.asarray(vals, float); idx = RNG.randint(0, len(a), (B, len(a))); m = a[idx].mean(1)
    return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(k, nm, sb, s) for k in ("unseen", "anytime") for sb in SIGMAS
            for nm in ORDER for s in SEEDS]
    raw = {k: {str(sb): {nm: [None] * len(SEEDS) for nm in ORDER} for sb in SIGMAS}
           for k in ("unseen", "anytime")}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            k, nm, sb, s, v = fut.result()
            raw[k][str(sb)][nm][s] = v
            done += 1
            if done % 24 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("choiceem8", {
        "meta": {"experiment": "ChoiceEM vs ChoiceCF vs RewardCF", "methods": ORDER,
                 "sigmas": SIGMAS, "rho": RHO, "seeds": SEEDS, "sigma_own": pc.SO,
                 "metric": "unseen + anytime, bootstrap CI"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# Choice-informativeness EM (joint latent + per-teammate gamma_k)\n",
         "Does EM-learned per-choice informativeness beat a fixed competence ramp? Sweep the "
         "broadcast REWARD noise sigma_obs (rho=1.0, own noise %.2f); 8 seeds, bootstrap 95%% CI.\n"
         % pc.SO]
    for k in ("unseen", "anytime"):
        L.append("## %s skill\n" % k.upper())
        L.append("| method | " + " | ".join("sigma_obs=%.1f" % s for s in SIGMAS) + " |")
        L.append("|" + "---|" * (len(SIGMAS) + 1))
        for nm in ORDER:
            cells = [cell(raw[k][str(sb)][nm]) for sb in SIGMAS]
            lab = "**%s**" % nm if nm == "ChoiceEM" else nm
            L.append("| %s | %s |" % (lab, " | ".join(cells)))
        L.append("")
    # ChoiceEM vs ChoiceCF deltas
    L.append("## ChoiceEM - ChoiceCF (mean; + = EM better)\n")
    L.append("| metric | " + " | ".join("sigma_obs=%.1f" % s for s in SIGMAS) + " |")
    L.append("|" + "---|" * (len(SIGMAS) + 1))
    for k in ("unseen", "anytime"):
        d = ["%+.3f" % (np.mean(raw[k][str(sb)]["ChoiceEM"]) - np.mean(raw[k][str(sb)]["ChoiceCF"]))
             for sb in SIGMAS]
        L.append("| %s | %s |" % (k, " | ".join(d)))
    L.append("")
    L.append("Read: if the choice channel is valuable (high sigma_obs), ChoiceEM should match or beat "
             "ChoiceCF by trusting model-based choices and discounting random ones; both choice methods "
             "should hold up better than RewardCF as rewards get noisy.\n")

    out_md = os.path.join(ROOT, "docs", "CHOICEEM.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
