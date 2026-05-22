"""
C14 (METHOD BAKE-OFF): all relevant competitors in ONE harness, on the natural
heterogeneous-MASKING regime (the C11 setting), with a GUESSED rank d_hat (fair).

Methods (grouped by structural assumption):
  Random              -- floor (learns nothing).                      [no-structure]
  UCBIndep            -- per-(drone,target) UCB1; correct heterogeneity,
                         but NO cross-arm generalization -> floor on unseen.
  UCBHomo             -- single shared arm table; pools across drones ->
                         confounded on heterogeneous worlds.          [no-structure]
  MFSGD               -- SGD matrix factorization (MF-CF).            [low-rank]
  ESTR                -- explore-then-spectral-refit (Kang-Hsieh-Lee 2022 style):
                         random explore, SVD of R_hat, then exploit.  [low-rank, centralized, explore-then-COMMIT]
  RewardCF (ours)     -- online weighted-ALS on own+others' noisy rewards.   [low-rank, online, decentralized]
  BothCF   (ours)     -- online weighted-ALS fusing rewards + competence-weighted choices.

Headline expectations:
  (1) UNSEEN-PAIR: every LOW-RANK method (MFSGD, ESTR, RewardCF, BothCF) lifts
      above the FLOOR; the no-structure methods (Random, UCBIndep, UCBHomo) sit
      AT the floor by construction. This is the categorical low-rank win.
  (2) AMONG low-rank methods: our ONLINE + DECENTRALIZED RewardCF/BothCF should
      be competitive-or-better than centralized explore-then-COMMIT ESTR,
      especially under masking (ESTR wastes its explore phase and never adapts).

Fairness: all structured learners get the SAME guessed rank d_hat (not true d).
Skill = (greedy - random) / (oracle - random), oracle = best-in-offered-subset.
Parallel across CPU cores (one process per (method, rho, seed) cell).
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

from pilot_c11_masking import run_masked
from pilot_noise import Tabular, RewardCF, BothCF, ChoiceCF, HybridCF, BothCFPrec, StackCF
from pilot_baselines import Random, UCBIndep, UCBHomo, MFSGD, ESTR, PTF, BPMF
from core import make_world
from _results_io import save_results

# ---- world / regime config (block model, signed cosine reward) ----
M, N, D, K = 30, 240, 5, 10        # K1=K2=K target/drone types -> eff. rank min(d,K)
D_HAT = 8                          # GUESSED rank for ALL structured learners (fair)
T, CAND = 50, 20                   # rounds, candidate-offer size
SO, SB = 0.10, 0.30               # own-obs noise, broadcast-obs noise
SEEDS = list(range(5))
RHOS = [1.0, 0.5, 0.25]            # per-drone fraction of broadcast observed

# ---- method registry: name -> (Class, hyperparams) ----
_ALS = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=8, refit_every=3)
_EPS = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)
REGISTRY = {
    "Random":   (Random,   {}),
    "UCBIndep": (UCBIndep, dict(c=2.0)),
    "UCBHomo":  (UCBHomo,  dict(c=2.0)),
    "MFSGD":    (MFSGD,    dict(lr=0.2, reg=1e-3, eps=0.15, init_scale=0.3)),
    "ESTR":     (ESTR,     dict(T_total=T, explore_frac=0.4, eps_floor=0.05)),
    "PTF":      (PTF,      dict(T_total=T, probe_frac=0.4, c=2.0, lr=0.2, reg=1e-3,
                               eps=0.15, init_scale=0.3)),
    "BPMF":     (BPMF,     dict(prior_var=1.0, init_scale=0.1)),
    "Tabular":  (Tabular,  dict(**_EPS)),
    "RewardCF": (RewardCF, dict(**_ALS)),
    "BothCF":   (BothCF,   dict(comp=True, s2c=0.2, n_neg=1, within=True,
                                warm_frac=0.3, T_total=T, **_ALS)),
    "ChoiceCF": (ChoiceCF, dict(comp=True, s2c=0.2, n_neg=1, within=True,
                                warm_frac=0.3, T_total=T, **_ALS)),
    # strict-ZK choice variant: GLOBAL negatives (within=False) -> needs only the
    # observed chosen action, not the teammate's offered menu (ZK audit, E13).
    "ChoiceZK": (ChoiceCF, dict(comp=True, s2c=0.2, n_neg=1, within=False,
                                warm_frac=0.3, T_total=T, **_ALS)),
    "HybridCF": (HybridCF, dict(T_total=T, probe_frac=0.3, probe_mode="ucb", c=2.0, **_ALS)),
    "BothCFPrec": (BothCFPrec, dict(comp=True, s2c=0.2, n_neg=1, within=True, gate_alpha=0.05,
                                    warm_frac=0.3, T_total=T, **_ALS)),
    "StackCF": (StackCF, dict(comp=True, s2c=0.2, n_neg=1, val_frac=0.3,
                              warm_frac=0.3, T_total=T, **_ALS)),
}
ORDER = ["Random", "UCBIndep", "UCBHomo", "Tabular",
         "MFSGD", "ESTR", "PTF", "BPMF", "RewardCF", "BothCF", "HybridCF"]
GROUP = {"Random": "no-struct", "UCBIndep": "no-struct", "UCBHomo": "no-struct",
         "Tabular": "no-struct", "MFSGD": "low-rank", "ESTR": "low-rank",
         "PTF": "low-rank", "BPMF": "low-rank",
         "RewardCF": "low-rank(ours)", "BothCF": "low-rank(ours)",
         "ChoiceCF": "low-rank(ours)", "HybridCF": "low-rank(ours)"}


def _run_cell(args):
    """One (method, rho, seed) cell -> (name, rho, seed, overall, unseen, uniq)."""
    name, rho, seed = args
    Cls, hp = REGISTRY[name]
    w = make_world(M, N, D, K, K, within=0.15, seed=seed, signed=True)
    o, u, q = run_masked(Cls, hp, w, T, seed, SO, SB, CAND, D_HAT, rho)
    return name, rho, seed, float(o), float(u), float(q)


def main():
    print("=" * 104)
    print("C14 METHOD BAKE-OFF on heterogeneous MASKING (block model, signed). "
          "m=%d n=%d d=%d(K=%d) d_hat=%d T=%d cand=%d %d seeds" %
          (M, N, D, K, D_HAT, T, CAND, len(SEEDS)))
    print("rho = per-drone fraction of broadcast observed (persistent). "
          "sigma_own=%.2f sigma_obs=%.2f" % (SO, SB))
    print("=" * 104)

    jobs = [(name, rho, s) for rho in RHOS for name in ORDER for s in SEEDS]
    # raw[rho][name] = dict(overall=[...], unseen=[...], uniq=[...])
    raw = {str(r): {nm: {"overall": [], "unseen": [], "uniq": []} for nm in ORDER}
           for r in RHOS}

    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_run_cell, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            name, rho, seed, o, u, q = fut.result()
            cell = raw[str(rho)][name]
            cell["overall"].append(o); cell["unseen"].append(u); cell["uniq"].append(q)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print("  ... %d/%d cells done" % (done, len(jobs)))

    for rho in RHOS:
        print("\n" + "-" * 104)
        print("rho = %.2f" % rho)
        print(f"{'method':>10s} {'group':>16s} | {'overall':>16s} | {'UNSEEN-pair':>16s} | {'stateUniq':>10s}")
        print("-" * 104)
        for nm in ORDER:
            c = raw[str(rho)][nm]
            om, os_ = np.mean(c["overall"]), np.std(c["overall"])
            um, us_ = np.mean(c["unseen"]), np.std(c["unseen"])
            uq = np.nanmean(c["uniq"])
            uqs = "%10.3f" % uq if np.isfinite(uq) else "%10s" % "-"
            print(f"{nm:>10s} {GROUP[nm]:>16s} | {om:7.3f} +- {os_:5.3f} | "
                  f"{um:7.3f} +- {us_:5.3f} | {uqs}")
    print("-" * 104)
    print("READ: UNSEEN-pair separates low-rank (lifts above ~0) from no-structure (floor).")
    print("Among low-rank: compare ESTR (centralized explore-then-commit) vs RewardCF/BothCF")
    print("(ours, online+decentralized). stateUniq>0 (CF/MF) => drones learn DIFFERENT models.")

    path = save_results("c14_compare", {
        "meta": {"experiment": "C14 method bake-off under heterogeneous masking",
                 "methods": ORDER, "group": GROUP,
                 "m": M, "n": N, "d": D, "K": K, "d_hat": D_HAT, "T": T, "cand": CAND,
                 "sigma_own": SO, "sigma_obs": SB, "seeds": SEEDS, "rhos": RHOS,
                 "metric": "per-seed overall/unseen skill + state-uniqueness",
                 "fairness": "all structured learners use guessed d_hat (not true d)"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
