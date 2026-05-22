"""
E7 (NEWCOMER COLD-START): a SECOND categorical result, the transpose of C12
onboarding. A drone that JOINS LATE with zero own history acts from the broadcast
alone: the swarm has exposed the target factors U; the newcomer recovers its OWN
factor p by a d-dim ridge FOLD-IN from k own probes, then predicts its WHOLE row,
including targets it never probed. Tabular newcomer knows only its k probed targets
(floor on the rest). Separation: CF needs O(d) own probes; tabular needs O(n).

Pipeline (ZK: newcomer uses only the public broadcast it passively observed for U,
plus its own k probe outcomes):
  1. Train m existing drones with RewardCF over the broadcast -> swarm-learned U_hat.
  2. Newcomer = a fresh latent row p_new ~ same distribution; give it k probes
     (random targets, own noisy rewards), fold-in p_hat = ridge(U_hat[probes], y),
     predict row = U_hat @ p_hat.
  3. Skill on UNSEEN (unprobed) targets vs k, CF vs Tabular vs random.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_noise import RewardCF
from core import make_world
from _results_io import save_results

PROBES = [0, 1, 2, 3, 5, 8, 16, 30]
SEEDS = list(range(10))
SO = pc.SO


def _skill_on(pred, r_row, pool, cand, rng, reps=200):
    g, o, b = [], [], []
    pool = np.asarray(pool)
    if len(pool) < cand:
        return float("nan")
    for _ in range(reps):
        off = rng.choice(pool, size=cand, replace=False)
        g.append(r_row[off[int(np.argmax(pred[off]))]]); o.append(r_row[off].max()); b.append(r_row[off].mean())
    gm, om, bm = np.mean(g), np.mean(o), np.mean(b)
    return float((gm - bm) / max(om - bm, 1e-6))


def run_newcomer(seed):
    m, n, d, K = pc.M, pc.N, pc.D, pc.K
    d_hat, T, cand = pc.D_HAT, pc.T, pc.CAND
    P, U, R, meta = make_world(m + 1, n, d, K, K, within=0.15, seed=seed, signed=True)
    # row m is the NEWCOMER; train existing rows 0..m-1
    als = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=10, refit_every=3)
    learners = [RewardCF(m, n, d_hat, i, seed + 7 * i + 1, **als) for i in range(m)]
    rng = np.random.RandomState(seed + 999)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):                       # full broadcast so the swarm exposes U well
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            for k in range(m):
                revealed[k] = true_r[k] + rng.normal(0, SO if k == i else pc.SB)
                rvar[k] = (SO if k == i else pc.SB) ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    U_hat = learners[0].U.copy()                 # swarm-learned target factors (n, d_hat)
    p_pop = np.mean([learners[0].P[i] for i in range(m)], axis=0)   # population mean factor (rank-1 prior)

    r_new = R[m]                                  # newcomer's true reward row
    g2 = np.random.RandomState(seed + 555)
    out = {"cf": {}, "tab": {}, "pop": {}}
    lam = 1.0 * np.eye(d_hat)                      # ridge toward the POPULATION prior (D7 shrinkage)
    pred_pop = U_hat @ p_pop                       # fixed popularity baseline
    for k in PROBES:
        probes = g2.choice(n, size=k, replace=False) if k > 0 else np.array([], int)
        y = r_new[probes] + g2.normal(0, SO, size=k) if k > 0 else np.array([])
        unseen = np.setdiff1d(np.arange(n), probes)
        # CF fold-in with prior mean p_pop: p = p_pop + (Up'Up + lam)^-1 Up'(y - Up p_pop).
        # k=0 -> p_pop (popularity); small k -> shrinks to population; k>=d -> own factor.
        if k >= 1:
            Up = U_hat[probes]
            p_hat = p_pop + np.linalg.solve(Up.T @ Up + lam, Up.T @ (y - Up @ p_pop))
            pred_cf = U_hat @ p_hat
        else:
            pred_cf = pred_pop                    # no own data -> population/popularity prior
        # Tabular newcomer: knows only probed targets; floor (0) on unprobed
        pred_tab = np.zeros(n)
        if k > 0:
            pred_tab[probes] = y
        out["cf"][k] = _skill_on(pred_cf, r_new, unseen, cand, g2)
        out["tab"][k] = _skill_on(pred_tab, r_new, unseen, cand, g2)
        out["pop"][k] = _skill_on(U_hat @ p_pop, r_new, unseen, cand, g2)
    return seed, out


def main():
    print("=" * 80)
    print("E7 NEWCOMER COLD-START: skill on UNSEEN targets vs # own probes "
          "(m=%d n=%d d=%d d_hat=%d, %d seeds)" % (pc.M, pc.N, pc.D, pc.D_HAT, len(SEEDS)))
    print("=" * 80)
    raw = {"cf": {str(k): [] for k in PROBES}, "tab": {str(k): [] for k in PROBES},
           "pop": {str(k): [] for k in PROBES}}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run_newcomer, s) for s in SEEDS]
        for fut in as_completed(futs):
            seed, out = fut.result()
            for k in PROBES:
                raw["cf"][str(k)].append(out["cf"][k])
                raw["tab"][str(k)].append(out["tab"][k])
                raw["pop"][str(k)].append(out["pop"][k])
    print("%6s | %10s %10s %10s" % ("probes", "CF foldin", "Tabular", "popularity"))
    print("-" * 44)
    for k in PROBES:
        cf = np.nanmean(raw["cf"][str(k)]); tb = np.nanmean(raw["tab"][str(k)]); pp = np.nanmean(raw["pop"][str(k)])
        print("%6d | %10.3f %10.3f %10.3f" % (k, cf, tb, pp))
    print("-" * 44)
    print("CATEGORICAL: CF reaches high newcomer skill at ~d_hat=%d probes (fold-in given U);" % pc.D_HAT)
    print("Tabular only learns its probed targets (floor on the unprobed ~n). Theta(d) vs Theta(n).")
    path = save_results("e7_newcomer", {
        "meta": {"experiment": "E7 newcomer cold-start (row fold-in given broadcast U)",
                 "probes": PROBES, "seeds": SEEDS, "m": pc.M, "n": pc.N, "d": pc.D,
                 "K": pc.K, "d_hat": pc.D_HAT, "T": pc.T, "cand": pc.CAND,
                 "sigma_own": SO, "sigma_obs": pc.SB,
                 "metric": "newcomer skill on UNSEEN (unprobed) targets vs #probes; CF foldin vs Tabular vs popularity"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
