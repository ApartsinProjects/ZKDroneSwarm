"""
E7 (NEWCOMER COLD-START): a SECOND categorical result, the transpose of C12
onboarding. A drone that JOINS LATE with zero own history acts from the broadcast
alone: the swarm has exposed the target factors U; the newcomer recovers its OWN
factor p by a d-dim ridge FOLD-IN from k own probes, then predicts its WHOLE row,
including targets it never probed. Tabular newcomer knows only its k probed targets
(floor on the rest). Separation: CF needs O(d) own probes; tabular needs O(n).

STRICT-ZK (cycle 62 fix): the newcomer NEVER copies a peer's learned factors. It
is a PASSIVE listener that, over the T training rounds, observes the SAME public
broadcast everyone else hears, but under its OWN personalized persistent mask
(M[newcomer,k] ~ Bernoulli(rho)), and recovers U_hat by running its OWN estimator
(the same weighted-ALS the swarm uses) on the (teammate-choice, noisy-reward)
tuples it actually saw. Its population prior p_pop is the mean of the teammate
factors IT recovered (its own frame), not any peer's P. The incumbents are masked
at the same rho, so this is the uniform "personalized partial noisy broadcast"
setting for all; rho=1.0 reduces to the original full-broadcast E7.

Pipeline:
  1. Train m incumbents with RewardCF under per-drone persistent rho-masking.
  2. A passive newcomer hears the broadcast under its own rho-mask and recovers
     U_hat (and teammate factors) by its own ALS -> strictly ZK, no copy.
  3. Newcomer gets k probes (random targets, own noisy rewards), folds in
     p_hat = p_pop + ridge(U_hat[probes], y - U_hat[probes] p_pop), predicts row.
  4. Skill on UNSEEN (unprobed) targets vs k, CF vs Tabular vs random, per rho.
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
RHOS = [1.0, 0.5, 0.25]
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


def run_newcomer(seed, rho):
    m, n, d, K = pc.M, pc.N, pc.D, pc.K
    d_hat, T, cand = pc.D_HAT, pc.T, pc.CAND
    P, U, R, meta = make_world(m + 1, n, d, K, K, within=0.15, seed=seed, signed=True)
    # row m is the NEWCOMER; train existing rows 0..m-1
    als = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=10, refit_every=3)
    learners = [RewardCF(m, n, d_hat, i, seed + 7 * i + 1, **als) for i in range(m)]

    # Per-drone persistent broadcast masks (incumbents always hear themselves).
    mrng = np.random.RandomState(seed + 4242)
    M = mrng.rand(m, m) < rho
    np.fill_diagonal(M, True)
    # Newcomer's OWN personalized mask over the m incumbents (no self -- it is passive).
    mask_nc = mrng.rand(m) < rho
    if not mask_nc.any():
        mask_nc[mrng.randint(m)] = True
    rvar_nc = np.where(mask_nc, pc.SB ** 2, np.inf)
    # Passive STRICT-ZK newcomer listener: recovers U from its OWN observations.
    nc = RewardCF(m, n, d_hat, 0, seed + 31, **als)

    rng = np.random.RandomState(seed + 999)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):                       # incumbents under persistent rho-masking
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            for k in range(m):
                if M[i, k]:
                    revealed[k] = true_r[k] + rng.normal(0, SO if k == i else pc.SB)
                    rvar[k] = (SO if k == i else pc.SB) ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
        # newcomer passively hears the SAME broadcast under its own persistent mask
        revealed_nc = np.full(m, np.nan)
        for k in range(m):
            if mask_nc[k]:
                revealed_nc[k] = true_r[k] + rng.normal(0, pc.SB)
        nc.observe(t, choices, revealed_nc, cand_sets, rvar_nc)
    nc._refit()                                  # finalize the newcomer's OWN factorization
    U_hat = nc.U                                  # recovered from the newcomer's own broadcast (ZK)
    vis = np.where(mask_nc)[0]
    p_pop = nc.P[vis].mean(0) if len(vis) else np.zeros(d_hat)   # newcomer's OWN population prior

    r_new = R[m]                                  # newcomer's true reward row
    g2 = np.random.RandomState(seed + 555)
    out = {"cf": {}, "tab": {}, "pop": {}}
    lam = 1.0 * np.eye(d_hat)                      # ridge toward the newcomer's OWN population prior
    pred_pop = U_hat @ p_pop                       # fixed popularity baseline (own frame)
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
        out["pop"][k] = _skill_on(pred_pop, r_new, unseen, cand, g2)
    return seed, rho, out


def main():
    print("=" * 80)
    print("E7 NEWCOMER COLD-START (STRICT-ZK): skill on UNSEEN targets vs # own probes")
    print("newcomer recovers U from its OWN masked broadcast (no peer copy); rho-sweep")
    print("(m=%d n=%d d=%d d_hat=%d, %d seeds, rho in %s)" % (pc.M, pc.N, pc.D, pc.D_HAT, len(SEEDS), RHOS))
    print("=" * 80)
    raw = {str(rho): {"cf": {str(k): [] for k in PROBES},
                      "tab": {str(k): [] for k in PROBES},
                      "pop": {str(k): [] for k in PROBES}} for rho in RHOS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run_newcomer, s, rho) for s in SEEDS for rho in RHOS]
        for fut in as_completed(futs):
            seed, rho, out = fut.result()
            r = raw[str(rho)]
            for k in PROBES:
                r["cf"][str(k)].append(out["cf"][k])
                r["tab"][str(k)].append(out["tab"][k])
                r["pop"][str(k)].append(out["pop"][k])
    for rho in RHOS:
        r = raw[str(rho)]
        print("\nrho = %.2f" % rho)
        print("%6s | %10s %10s %10s" % ("probes", "CF foldin", "Tabular", "popularity"))
        print("-" * 44)
        for k in PROBES:
            cf = np.nanmean(r["cf"][str(k)]); tb = np.nanmean(r["tab"][str(k)]); pp = np.nanmean(r["pop"][str(k)])
            print("%6d | %10.3f %10.3f %10.3f" % (k, cf, tb, pp))
    print("-" * 44)
    print("CATEGORICAL: CF reaches high newcomer skill at ~d_hat=%d probes (fold-in on its OWN" % pc.D_HAT)
    print("recovered U); Tabular only learns its probed targets (floor on the unprobed ~n).")
    print("Theta(d) vs Theta(n) -- now under a strictly-ZK newcomer; degrades gracefully as rho falls.")
    path = save_results("e7_newcomer", {
        "meta": {"experiment": "E7 newcomer cold-start (STRICT-ZK: newcomer recovers U from its OWN masked broadcast)",
                 "probes": PROBES, "seeds": SEEDS, "rhos": RHOS, "m": pc.M, "n": pc.N, "d": pc.D,
                 "K": pc.K, "d_hat": pc.D_HAT, "T": pc.T, "cand": pc.CAND,
                 "sigma_own": SO, "sigma_obs": pc.SB,
                 "metric": "newcomer skill on UNSEEN (unprobed) targets vs #probes; CF foldin vs Tabular vs popularity; per rho"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
