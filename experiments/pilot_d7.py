"""D7 / H8: hierarchical / TYPE-PRIOR shrinkage for cold-start. E7's newcomer folds in its factor by
shrinking toward the POPULATION mean. D7/H8 asks: if the swarm has latent TYPES (clusters), does
shrinking toward the newcomer's own TYPE centroid (inferred ZK from its few probes) beat shrinking to
the global mean, especially at very small probe budgets? Strictly ZK: the newcomer recovers U + the
teammate factors from its OWN passive broadcast, k-means-clusters the recovered teammate factors into
K types, infers its type from its probes (best-fitting centroid), and shrinks to that centroid.
Compares CF-pop (population prior, = E7) vs CF-type (type prior) vs Tabular on unseen skill vs #probes.
Writes docs/D7.md."""
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from pilot_noise import RewardCF
from core import make_world
from pilot_e7_newcomer import _skill_on
from _results_io import save_results

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBES = [1, 2, 3, 5, 8, 16]
SEEDS = list(range(10))
SO = pc.SO


def _kmeans(X, K, iters=20, seed=0):
    rng = np.random.RandomState(seed)
    C = X[rng.choice(len(X), size=min(K, len(X)), replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        Cn = np.array([X[a == k].mean(0) if np.any(a == k) else C[k] for k in range(len(C))])
        if np.allclose(Cn, C):
            break
        C = Cn
    return C


def run_d7(seed):
    m, n, d, K = pc.M, pc.N, pc.D, pc.K
    d_hat, T, cand = pc.D_HAT, pc.T, pc.CAND
    P, U, R, meta = make_world(m + 1, n, d, K, K, within=0.15, seed=seed, signed=True)
    als = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=10, refit_every=3)
    learners = [RewardCF(m, n, d_hat, i, seed + 7 * i + 1, **als) for i in range(m)]
    rng = np.random.RandomState(seed + 999)
    nc = RewardCF(m, n, d_hat, 0, seed + 31, **als)            # passive ZK listener (recovers U + P)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            for k in range(m):
                revealed[k] = true_r[k] + rng.normal(0, SO if k == i else pc.SB)
                rvar[k] = (SO if k == i else pc.SB) ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
        revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
        for k in range(m):
            revealed[k] = true_r[k] + rng.normal(0, pc.SB); rvar[k] = pc.SB ** 2
        nc.observe(t, choices, revealed, cand_sets, rvar)
    nc._refit()
    U_hat = nc.U
    p_pop = nc.P.mean(0)                                       # population prior (E7)
    cents = _kmeans(nc.P, K, seed=seed)                        # TYPE centroids from recovered teammate factors

    r_new = R[m]; g2 = np.random.RandomState(seed + 555)
    lam = 1.0 * np.eye(d_hat)
    out = {"pop": {}, "type": {}, "tab": {}}
    for k in PROBES:
        probes = g2.choice(n, size=k, replace=False)
        y = r_new[probes] + g2.normal(0, SO, size=k)
        unseen = np.setdiff1d(np.arange(n), probes)
        Up = U_hat[probes]
        # population-prior fold-in (E7)
        p_pop_hat = p_pop + np.linalg.solve(Up.T @ Up + lam, Up.T @ (y - Up @ p_pop))
        # type-prior fold-in: pick the centroid whose prior-fold-in best fits the probes, shrink to it
        best_c, best_err = p_pop, np.inf
        for c in cents:
            pc_hat = c + np.linalg.solve(Up.T @ Up + lam, Up.T @ (y - Up @ c))
            err = float(np.sum((Up @ pc_hat - y) ** 2) + 0.1 * np.sum((pc_hat - c) ** 2))
            if err < best_err:
                best_err, best_c = err, c
        p_type_hat = best_c + np.linalg.solve(Up.T @ Up + lam, Up.T @ (y - Up @ best_c))
        pred_tab = np.zeros(n); pred_tab[probes] = y
        out["pop"][k] = _skill_on(U_hat @ p_pop_hat, r_new, unseen, cand, g2)
        out["type"][k] = _skill_on(U_hat @ p_type_hat, r_new, unseen, cand, g2)
        out["tab"][k] = _skill_on(pred_tab, r_new, unseen, cand, g2)
    return seed, out


def main():
    raw = {kk: {str(k): [] for k in PROBES} for kk in ("pop", "type", "tab")}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(run_d7, s) for s in SEEDS]
        done = 0
        for fut in as_completed(futs):
            seed, out = fut.result()
            for kk in ("pop", "type", "tab"):
                for k in PROBES:
                    raw[kk][str(k)].append(out[kk][k])
            done += 1
            if done % 3 == 0 or done == len(SEEDS):
                print("  ... %d/%d seeds" % (done, len(SEEDS)))

    save_results("d7_typeprior", {
        "meta": {"experiment": "D7/H8 type-prior vs population-prior shrinkage for cold-start newcomer",
                 "probes": PROBES, "seeds": SEEDS, "m": pc.M, "n": pc.N, "d": pc.D, "K": pc.K,
                 "d_hat": pc.D_HAT, "metric": "newcomer unseen skill vs #probes; type-prior vs pop-prior vs tabular"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# D7/H8: type-prior vs population-prior shrinkage for cold-start\n",
         "Newcomer unseen-pair skill vs #own probes; CF-type shrinks to the inferred TYPE centroid, "
         "CF-pop to the global mean (E7). 10 seeds.\n",
         "| probes | CF-type | CF-pop (E7) | Tabular |", "|---|---|---|---|"]
    for k in PROBES:
        L.append("| %d | %.3f | %.3f | %.3f |" % (k, np.nanmean(raw["type"][str(k)]),
                                                  np.nanmean(raw["pop"][str(k)]), np.nanmean(raw["tab"][str(k)])))
    L.append("")
    deltas = ["k=%d: %+.3f" % (k, np.nanmean(raw["type"][str(k)]) - np.nanmean(raw["pop"][str(k)])) for k in PROBES]
    L.append("Type-prior minus population-prior (advantage of hierarchical shrinkage): " + ";  ".join(deltas) + "\n")
    L.append("Read: if CF-type beats CF-pop most at SMALL probe budgets (k<=d), the type centroid is a "
             "better prior than the global mean when own data is scarce; the gap should vanish as k grows "
             "(both converge to the own factor). Both should beat Tabular (floor on unseen).\n")
    out_md = os.path.join(ROOT, "docs", "D7.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
