"""
C12: TWO-PHASE dynamic TARGET ONBOARDING (reward-observable).

Phase 1: all drones learn the drone factors P (who is similar) from the shared
reward broadcast over an INITIAL target set (ALS; p=1 so the broadcast yields one
consistent P_hat each drone computes).
Phase 2: NEW targets are injected; for each, `probes` drones pull it once (quick
joint sample, reward broadcast); u_hat_j is fit by a d-dim RIDGE projection given
the known P_hat (= WALS/fold-in item cold-start); then ALL drones predict it.

Separation: TABULAR needs EVERY drone to probe EVERY new target (a drone knows a
new target only if it personally probed it) -> Theta(m) probes/target for full
coverage. CF needs ~Theta(d) shared probes/target (enough to identify the d-dim
u_j) and then ALL drones predict. Sweep `probes`: CF saturates near d_hat;
tabular rises ~linearly to probes=m.

Metric: skill on NEW targets = (greedy - rand)/(oracle - rand), per drone,
offering subsets of NEW targets. Fair: CF uses guessed rank d_hat.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from core import make_world
from _results_io import save_results


def als_fit(K, J, V, m, n, d, sweeps=15, lam=1.0, seed=0):
    rng = np.random.RandomState(seed)
    P = rng.normal(0, 0.1, (m, d)); U = rng.normal(0, 0.1, (n, d))
    I = lam * np.eye(d)
    K = np.asarray(K); J = np.asarray(J); V = np.asarray(V)
    for _ in range(sweeps):
        Pk = P[K]
        AU = np.tile(I, (n, 1, 1)).astype(float); bU = np.zeros((n, d))
        np.add.at(AU, J, np.einsum('ni,nj->nij', Pk, Pk)); np.add.at(bU, J, V[:, None] * Pk)
        U = np.linalg.solve(AU, bU[..., None])[..., 0]
        Uj = U[J]
        AP = np.tile(I, (m, 1, 1)).astype(float); bP = np.zeros((m, d))
        np.add.at(AP, K, np.einsum('ni,nj->nij', Uj, Uj)); np.add.at(bP, K, V[:, None] * Uj)
        P = np.linalg.solve(AP, bP[..., None])[..., 0]
    return P, U


def experiment(world, n0, probes, T1, d_hat, sigma, seed, cand=10):
    P, U, R = world[:3]; m, n = R.shape
    rng = np.random.RandomState(seed + 13)
    # Phase 1: collect obs on INITIAL targets (each drone pulls T1 random initial targets)
    K, J, V = [], [], []
    for k in range(m):
        for _ in range(T1):
            j = rng.randint(n0)
            K.append(k); J.append(j); V.append(R[k, j] + rng.normal(0, sigma))
    P_hat, _ = als_fit(K, J, V, m, n0, d_hat, sweeps=15, lam=1.0, seed=seed)
    own_mean = np.mean(V)  # tabular default for unknown
    # Phase 2: onboard NEW targets (indices n0..n-1)
    new = np.arange(n0, n)
    u_hat = {}
    tab_known = {}                       # (drone, new_target) -> observed reward
    I = 1.0 * np.eye(d_hat)
    for j in new:
        probers = rng.choice(m, size=min(probes, m), replace=False)
        r = np.array([R[k, j] + rng.normal(0, sigma) for k in probers])
        Pp = P_hat[probers]
        u_hat[j] = np.linalg.solve(Pp.T @ Pp + I, Pp.T @ r)   # ridge / fold-in
        for k, rk in zip(probers, r):
            tab_known[(k, j)] = rk
    # Eval on NEW targets: offer subsets of new targets, greedy per drone
    g = np.random.RandomState(seed + 555)
    cf_g, tb_g, oc, rd = [], [], [], []
    for _ in range(150):
        for k in range(m):
            off = g.choice(new, size=cand, replace=False)
            cf_scores = np.array([P_hat[k] @ u_hat[j] for j in off])
            cf_g.append(R[k, off[int(np.argmax(cf_scores))]])
            tb_scores = np.array([tab_known.get((k, j), own_mean) for j in off])
            tb_g.append(R[k, off[int(np.argmax(tb_scores))]])
            oc.append(R[k, off].max()); rd.append(R[k, off].mean())
    ocm, rdm = np.mean(oc), np.mean(rd); den = max(ocm - rdm, 1e-6)
    return (np.mean(cf_g) - rdm) / den, (np.mean(tb_g) - rdm) / den


def main():
    m, n, d, K = 30, 260, 5, 10
    n0 = 200; d_hat = 8; T1 = 60; sigma = 0.10; cand = 10
    seeds = list(range(5))
    probes_list = [1, 2, 3, 5, 8, 15, 30]
    print("=" * 80)
    print("C12: dynamic TARGET ONBOARDING (block model). m=%d n0=%d new=%d d=%d(K=%d) "
          "d_hat=%d T1=%d %d seeds" % (m, n0, n - n0, d, K, d_hat, T1, len(seeds)))
    print("skill on NEW targets vs #probes/target. CF=ridge fold-in given P; "
          "Tabular=knows only self-probed.")
    print("=" * 80)
    print(f"{'probes/target':>13s} | {'CF skill':>9s} | {'Tabular skill':>13s} | {'CF-Tab':>7s}")
    print("-" * 80)
    raw = {}
    for probes in probes_list:
        cf, tb = [], []
        for s in seeds:
            w = make_world(m, n, d, K, K, within=0.15, seed=s, signed=True)
            c, t = experiment(w, n0, probes, T1, d_hat, sigma, s, cand)
            cf.append(c); tb.append(t)
        raw[str(probes)] = dict(cf=cf, tab=tb)
        print(f"{probes:13d} | {np.mean(cf):9.3f} | {np.mean(tb):13.3f} | {np.mean(cf)-np.mean(tb):+7.3f}")
    print("-" * 80)
    print("EXPECT: CF saturates near probes ~ d_hat (few SHARED probes onboard a new")
    print("target for ALL drones); Tabular rises ~linearly, needing probes ~ m. => Theta(d) vs Theta(m).")
    path = save_results("c12_onboard", {
        "meta": {"experiment": "C12 dynamic target onboarding", "m": m, "n": n,
                 "n0": n0, "n_new": n - n0, "d": d, "K": K, "d_hat": d_hat, "T1": T1,
                 "sigma": sigma, "cand": cand, "seeds": seeds, "probes_list": probes_list,
                 "metric": "per-seed skill on NEW targets (CF vs Tabular)"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
