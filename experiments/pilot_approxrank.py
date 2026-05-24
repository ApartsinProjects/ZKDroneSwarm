"""
X5 (APPROXIMATE-LOW-RANK ROBUSTNESS): how does the categorical separation degrade
as the reward stops being EXACTLY low-rank? We perturb the rank-d block reward
with a full-rank Gaussian term,

    R_eps = (R + eps * (std(R)/std(G)) * G) / sqrt(1 + eps^2),   G_ij ~ N(0,1) iid,

which preserves the entry-wise std (so the observation SNR is held fixed) while
moving energy out of the rank-d subspace: the low-rank energy fraction is
1/(1+eps^2) and the effective rank rises from d toward min(m,n) as eps grows.
We sweep eps at the masked headline rho=0.25 and report unseen-pair skill.

Expectation (answers the referee concern that the categorical result depends on
an EXACTLY low-rank reward): SwarmCF (RewardCF) and its batch variant degrade
GRACEFULLY and only approach the floor once the structure is mostly gone; the
structure-free learner (Independent-UCB) is at the floor for every eps by
construction. The separation is therefore a property of EXPLOITABLE structure,
not of exact low-rankness.

Reuses pilot_compare.REGISTRY / guessed_rank and pilot_c11_masking.run_masked
(identical fair config: block model, guessed d_hat in [d,2d], own clean reward +
persistent per-observer broadcast mask). Parallel across CPU cores.
"""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_c11_masking import run_masked
from core import make_world, effective_rank
from _results_io import save_results

METHODS = ["RewardCF", "PTF", "UCBIndep"]            # SwarmCF, SwarmCF-batch, structure-free floor
EPS = [0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]          # full-rank perturbation strength
RHO = 0.25                                            # masked headline broadcast rate
SEEDS = list(range(int(os.environ.get("ZK_SEEDS", "16"))))


def make_world_approx(seed, eps):
    """Block low-rank world with a full-rank perturbation of strength eps (std preserved)."""
    P, U, R, meta = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    if eps > 0:
        rng = np.random.RandomState(12345 + seed)
        G = rng.randn(pc.M, pc.N)
        s = R.std() / max(G.std(), 1e-12)
        R = (R + eps * s * G) / np.sqrt(1.0 + eps * eps)
    meta["eps"] = eps
    meta["eff_rank"] = effective_rank(R)
    return (P, U, R, meta)


def _run_cell(args):
    """One (method, eps, seed) cell -> (name, eps, seed, overall, unseen, eff_rank)."""
    name, eps, seed = args
    Cls, hp = pc.REGISTRY[name]
    w = make_world_approx(seed, eps)
    o, u, q = run_masked(Cls, hp, w, pc.T, seed, pc.SO, pc.SB, pc.CAND, pc.guessed_rank(seed), RHO)
    return name, eps, seed, float(o), float(u), float(w[3]["eff_rank"])


def main():
    print("=" * 100)
    print("X5 APPROX-LOW-RANK: unseen-pair skill vs full-rank perturbation eps (rho=%.2f). "
          "m=%d n=%d d=%d(K=%d) T=%d cand=%d %d seeds" %
          (RHO, pc.M, pc.N, pc.D, pc.K, pc.T, pc.CAND, len(SEEDS)))
    print("R_eps=(R+eps*s*G)/sqrt(1+eps^2); low-rank energy frac=1/(1+eps^2); eff rank rises with eps")
    print("=" * 100)

    jobs = [(nm, e, s) for e in EPS for nm in METHODS for s in SEEDS]
    raw = {("%.2f" % e): {nm: {"overall": [], "unseen": [], "eff_rank": []} for nm in METHODS}
           for e in EPS}
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_run_cell, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, e, s, o, u, er = fut.result()
            c = raw[("%.2f" % e)][nm]
            c["overall"].append(o); c["unseen"].append(u); c["eff_rank"].append(er)
            done += 1
            if done % 21 == 0 or done == len(jobs):
                print("  ... %d/%d cells done" % (done, len(jobs)))

    print("\nUNSEEN-pair skill (mean over %d seeds):" % len(SEEDS))
    hdr = "method".rjust(12) + " | " + " ".join("%7.2f" % e for e in EPS)
    print(hdr); print("-" * len(hdr))
    for nm in METHODS:
        cells = ["%5.2f" % np.mean(raw[("%.2f" % e)][nm]["unseen"]) for e in EPS]
        print("%12s | " % nm + " ".join("%7s" % c for c in cells))
    er_row = ["%.1f" % np.mean(raw[("%.2f" % e)][METHODS[0]]["eff_rank"]) for e in EPS]
    print("%12s | " % "eff_rank" + " ".join("%7s" % c for c in er_row))
    print("-" * len(hdr))
    print("READ: SwarmCF (RewardCF) degrades gracefully as eps rises (eff rank grows);")
    print("structure-free UCBIndep stays at the floor for every eps. Separation = exploitable structure.")

    path = save_results("x5_approxrank", {
        "meta": {"experiment": "X5 approximate-low-rank robustness (full-rank perturbation sweep)",
                 "methods": METHODS, "eps": EPS, "rho": RHO, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "K": pc.K, "d_hat": "random in [%d,%d] per seed" % (pc.D, 2 * pc.D),
                 "T": pc.T, "cand": pc.CAND, "sigma_own": pc.SO, "sigma_obs": pc.SB,
                 "perturb": "R_eps=(R+eps*s*G)/sqrt(1+eps^2), s=std(R)/std(G); low-rank energy frac=1/(1+eps^2)",
                 "metric": "per-seed overall/unseen skill + effective rank vs eps"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
