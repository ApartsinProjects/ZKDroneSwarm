"""
H3 capstone validation: does ONE method (UnifiedCF) match the per-regime SPECIALISTS across
the whole design space? UnifiedCF = EMCF (confidence + predictive-variance UCB + ARD-capable)
UNITED with a loss-self-gating de-confliction offset and a loss-gated exploration anneal, so it
is pure EMCF when the drone never loses and de-conflicts only under real contention.

On the SAME 8 seeds we compare UnifiedCF to the specialist that wins each regime:
  STANDARD (rho=1, no contention): anytime skill        vs EMCF (the confidence specialist)
  CHURN (continuous turnover):     active + recent skill vs EMCF (H6b winner)
  CONTENTION pool=15 (severe):     earned reward         vs ContentionAdaCF + greedy RewardCFconv
  CONTENTION pool=240 (none):      earned reward         vs ContentionAdaCF + greedy RewardCFconv
Claim: UnifiedCF is best-or-statistically-tied EVERYWHERE. 8 seeds, bootstrap 95% CI.
Writes docs/UNIFIED.md. Saves raw first.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_anytime as pa
import pilot_contention as pcon
import pilot_churn as pch
from pilot_noise import UnifiedCF, EMCF, ContentionAdaptiveCF
from core import make_world
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = list(range(8))
RHO = 1.0

UNI = dict(em_beta=1.5, em_sweeps=8, refit_every=2, eps_hi=0.8, lr=0.15, coll_pow=2.0,
           beta_anneal=0.4, eps0=0.5, eps_min=0.05, eps_decay=0.97)
EM = dict(em_beta=1.5, em_sweeps=8, refit_every=2, eps0=0.5, eps_min=0.05, eps_decay=0.97)
ADA = pcon.REG["ContentionAdaCF"][1]
GREEDY = pcon.REG["RewardCFconv"][1]
RNG = np.random.RandomState(0)


def _job(seed):
    o = {}
    o["std_any_uni"] = pa.run_anytime_clshp(UnifiedCF, UNI, RHO, seed)[-1]
    o["std_any_em"] = pa.run_anytime_clshp(EMCF, EM, RHO, seed)[-1]
    o["chu_act_uni"], o["chu_rec_uni"] = pch._run(UnifiedCF, UNI, seed)
    o["chu_act_em"], o["chu_rec_em"] = pch._run(EMCF, EM, seed)
    for pool in (15, 240):
        o["con%d_uni" % pool] = pcon.run_contention(UnifiedCF, UNI, RHO, pool, seed)[0]
        o["con%d_ada" % pool] = pcon.run_contention(ContentionAdaptiveCF, ADA, RHO, pool, seed)[0]
        o["con%d_grd" % pool] = pcon.run_contention(*pcon.REG["RewardCFconv"], RHO, pool, seed)[0]
    return seed, o


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v is not None], float)
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    keys = ["std_any_uni", "std_any_em", "chu_act_uni", "chu_rec_uni", "chu_act_em", "chu_rec_em",
            "con15_uni", "con15_ada", "con15_grd", "con240_uni", "con240_ada", "con240_grd"]
    raw = {k: [None] * len(SEEDS) for k in keys}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, s): s for s in SEEDS}
        done = 0
        for fut in as_completed(futs):
            s, o = fut.result()
            for k in keys:
                raw[k][s] = o.get(k)
            done += 1
            print("  ... %d/%d seeds" % (done, len(SEEDS)))

    save_results("unified8", {
        "meta": {"experiment": "H3 UnifiedCF capstone validation across regimes",
                 "seeds": SEEDS, "rho": RHO, "uni_hp": {k: v for k, v in UNI.items()},
                 "m": pc.M, "n": pc.N, "d": pc.D, "d_hat": pc.D_HAT},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# H3: UnifiedCF capstone validation (one method, best-or-tied across regimes)\n",
         "UnifiedCF (EMCF + loss-self-gating de-confliction offset + loss-gated exploration anneal) vs "
         "the per-regime specialist, SAME 8 seeds, bootstrap 95%% CI. rho=%.1f.\n" % RHO]
    L.append("## STANDARD (no contention): anytime skill\n")
    L.append("| UnifiedCF | EMCF (specialist) |\n|---|---|")
    L.append("| %s | %s |\n" % (cell(raw["std_any_uni"]), cell(raw["std_any_em"])))
    L.append("## CHURN: active-set / recent-arrival skill\n")
    L.append("| metric | UnifiedCF | EMCF (specialist) |\n|---|---|---|")
    L.append("| active | %s | %s |" % (cell(raw["chu_act_uni"]), cell(raw["chu_act_em"])))
    L.append("| recent | %s | %s |\n" % (cell(raw["chu_rec_uni"]), cell(raw["chu_rec_em"])))
    L.append("## CONTENTION: earned-reward skill\n")
    L.append("| pool | UnifiedCF | ContentionAdaCF (specialist) | greedy RewardCFconv |\n|---|---|---|---|")
    for pool in (15, 240):
        L.append("| %d | %s | %s | %s |" % (pool, cell(raw["con%d_uni" % pool]),
                                            cell(raw["con%d_ada" % pool]), cell(raw["con%d_grd" % pool])))
    L.append("")
    L.append("Read: UnifiedCF should TIE EMCF on standard+churn (it reduces to EMCF when the drone "
             "never loses) and TIE-or-beat ContentionAdaCF / greedy under contention (it self-engages "
             "the offset and anneals exploration when it starts losing). One method, no per-regime "
             "tuning, best-or-statistically-tied everywhere = the design space collapses to one method.\n")

    out_md = os.path.join(ROOT, "docs", "UNIFIED.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
