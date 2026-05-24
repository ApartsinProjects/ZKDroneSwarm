"""PARITY CHECK (3-seed, in-process, permitted): compare the NEW latentswarm package path against
the analytical harness experiments/pilot_c11_masking.run_masked on identical seeds/params.

Params (per the task): block_cosine, m=30, n=240, d=5, n_types=10, T=50, offer_size=20, rho=0.5,
sigma_own=0.10, sigma_obs=0.30, d_hat = RandomState(9000+seed).randint(5, 11). seeds {0,1,2}.

Metrics compared: SwarmCF overall + unseen skill, Tabular unseen, SwarmCF state-uniqueness.
This is NOT a full rerun -- it is the acceptance test (3 seeds) the developer guide prescribes.
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)                                # latentswarm package
sys.path.insert(0, os.path.join(ROOT, "experiments"))  # pilot_* analytical harness

# --- analytical harness (the reference) ---
from pilot_c11_masking import run_masked
from pilot_noise import RewardCF as PilotRewardCF, Tabular as PilotTabular
from core import make_world

# --- new latentswarm path ---
from latentswarm.config import RunConfig
from latentswarm.sweeps import run_cell, base_config, _cfg_for

SEEDS = [0, 1, 2]
M, N, D, K = 30, 240, 5, 10
T, CAND = 50, 20
SO, SB, RHO = 0.10, 0.30, 0.5

# analytical-harness hyperparameters (from experiments/pilot_compare)
_ALS = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=8, refit_every=3)
_EPS = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)


def dhat(seed):
    return int(np.random.RandomState(9000 + seed).randint(D, 2 * D + 1))


def analytical(seed):
    w = make_world(M, N, D, K, K, within=0.15, seed=seed, signed=True)
    dh = dhat(seed)
    cf_ov, cf_un, cf_uq = run_masked(PilotRewardCF, _ALS, w, T, seed, SO, SB, CAND, dh, RHO)
    tb_ov, tb_un, tb_uq = run_masked(PilotTabular, _EPS, w, T, seed, SO, SB, CAND, dh, RHO)
    return dict(cf_overall=cf_ov, cf_unseen=cf_un, cf_uniq=cf_uq, tab_unseen=tb_un, dhat=dh)


def latentswarm(seed):
    cfg = base_config(  # the headline regime; force the exact parity params
        m=M, n=N, d=D, T=T, n_types=K, offer_size=CAND, scenario="block_cosine", jitter=0.15,
        mask_mode="persistent", rho=RHO, sigma_own=SO, sigma_obs=SB,
        als_sweeps=8, refit_every=3, epsilon=0.5, epsilon_decay=0.93, epsilon_min=0.05,
        rank_guess="random", rank_lo=D, rank_hi=2 * D, seeds=SEEDS)
    cf = run_cell(cfg, "swarm_cf", seed)
    tb = run_cell(cfg, "tabular", seed)
    return dict(cf_overall=cf["overall"], cf_unseen=cf["unseen"], cf_uniq=cf["uniq"],
                tab_unseen=tb["unseen"], dhat=dhat(seed))


def main():
    rows = []
    for s in SEEDS:
        a = analytical(s); l = latentswarm(s)
        assert a["dhat"] == l["dhat"], (a["dhat"], l["dhat"])
        rows.append((s, a, l))

    def col(rows, who, key):
        return np.array([r[1 if who == "ana" else 2][key] for r in rows], float)

    print("=" * 96)
    print("PARITY CHECK: latentswarm (block_cosine path) vs analytical run_masked  |  m=%d n=%d d=%d "
          "K=%d T=%d c=%d rho=%.2f" % (M, N, D, K, T, CAND, RHO))
    print("sigma_own=%.2f sigma_obs=%.2f  d_hat=RandomState(9000+seed).randint(%d,%d)  seeds=%s"
          % (SO, SB, D, 2 * D + 1, SEEDS))
    print("=" * 96)
    metrics = [("SwarmCF overall", "cf_overall"), ("SwarmCF unseen", "cf_unseen"),
               ("Tabular unseen", "tab_unseen"), ("SwarmCF stateUniq", "cf_uniq")]
    print(f"{'metric':>18s} {'seed':>5s} | {'analytical':>12s} {'latentswarm':>12s} {'diff':>9s}")
    print("-" * 96)
    for label, key in metrics:
        for (s, a, l) in rows:
            av, lv = a[key], l[key]
            print(f"{label:>18s} {s:>5d} | {av:12.4f} {lv:12.4f} {lv-av:+9.4f}")
        av = col(rows, "ana", key); lv = col(rows, "ls", key)
        print(f"{label:>18s} {'mean':>5s} | {av.mean():12.4f} {lv.mean():12.4f} {lv.mean()-av.mean():+9.4f}"
              f"   (ana sd={av.std():.4f}, ls sd={lv.std():.4f})")
        print("-" * 96)

    print("\nVERDICT (statistical parity = means within ~1 std / overlapping ranges):")
    ok_all = True
    for label, key in metrics:
        av = col(rows, "ana", key); lv = col(rows, "ls", key)
        pooled_sd = max(np.sqrt(0.5 * (av.std() ** 2 + lv.std() ** 2)), 1e-6)
        within = abs(av.mean() - lv.mean()) <= max(av.std(), lv.std(), 0.02) + 1e-9
        # range overlap
        a_lo, a_hi = av.min(), av.max(); l_lo, l_hi = lv.min(), lv.max()
        overlap = not (a_hi < l_lo or l_hi < a_lo)
        verdict = "PARITY" if (within or overlap) else "DIVERGE"
        ok_all = ok_all and (within or overlap)
        print("  %-18s mean diff %+.4f  (|diff|/maxsd=%.2f)  ranges %s  -> %s"
              % (label, lv.mean() - av.mean(), abs(av.mean() - lv.mean()) / pooled_sd,
                 "overlap" if overlap else "disjoint", verdict))
    print("\nOVERALL:", "STATISTICAL PARITY HOLDS" if ok_all else "SOME METRICS DIVERGE (see caveats)")


if __name__ == "__main__":
    main()
