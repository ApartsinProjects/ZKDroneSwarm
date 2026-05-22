"""
E12 (MASKING MODEL): repeat the headline metrics under IID per-round loss vs the
PERSISTENT mask, to test Theorem 4.
  PERSISTENT: M_{ik} ~ Bernoulli(rho) once, fixed for all rounds.
  IID:        M^t_{ik} ~ Bernoulli(rho) independently each round.
Both hide BOTH channels consistently (masked teammate -> neither choice nor reward).
Metrics per cell: final-policy UNSEEN skill, ANYTIME cumulative skill, and
state-UNIQUENESS (divergence of drones' learned reward matrices; defined for the
factor methods).
Theorem 4 predictions: (c) unseen + anytime are ~INVARIANT to the masking model;
(a/b) state-uniqueness is DURABLE under persistent but TRANSIENT under iid (lower
at fixed T, and decreasing in T) -- tested in Part B (uniqueness vs T).
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from core import make_world
from _results_io import save_results

METHODS = ["Tabular", "UCBIndep", "PTF", "ESTR", "RewardCF", "HybridCF"]
MODES = ["persistent", "iid"]
RHOS = [1.0, 0.5, 0.25, 0.1]
SEEDS = list(range(8))
TGRID = [25, 50, 100, 200]          # Part B: uniqueness vs horizon


def _state_uniq(learners, m):
    L0 = learners[0]
    P0 = getattr(L0, "P", None); U0 = getattr(L0, "U", None)
    if P0 is None or U0 is None:
        return float("nan")
    try:
        Rh = np.array([(learners[i].P @ learners[i].U.T).ravel() for i in range(m)])
    except Exception:
        return float("nan")
    Rh = Rh - Rh.mean(1, keepdims=True)
    nrm = np.linalg.norm(Rh, axis=1, keepdims=True) + 1e-9
    cos = (Rh / nrm) @ (Rh / nrm).T
    return 1.0 - float(cos[np.triu_indices(m, 1)].mean())


def run_iid(args):
    name, mode, rho, seed, T = args
    Cls, base = pc.REGISTRY[name]; hp = dict(base)
    if "T_total" in hp: hp["T_total"] = T
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m, n = R.shape
    cand, so, sb = pc.CAND, pc.SO, pc.SB
    rng = np.random.RandomState(seed + 999)
    Mfix = (rng.rand(m, m) < rho); np.fill_diagonal(Mfix, True)   # persistent mask
    learners = [Cls(m, n, pc.D_HAT, i, seed + 7 * i + 1, **hp) for i in range(m)]
    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        real[t] = true_r.sum()
        orac[t] = sum(R[i, cand_sets[i]].max() for i in range(m))
        rnd[t] = sum(R[i, cand_sets[i]].mean() for i in range(m))
        if mode == "iid":
            Mt = (rng.rand(m, m) < rho); np.fill_diagonal(Mt, True)
        else:
            Mt = Mfix
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf); ch_i = choices.copy()
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i:
                    if Mt[i, k]:
                        revealed[k] = true_r[k] + rng.normal(0, sb); rvar[k] = sb ** 2
                    else:
                        ch_i[k] = -1
            learners[i].observe(t, ch_i, revealed, cand_sets, rvar)
    preds = [learners[i].predict_scores() for i in range(m)]
    pulled = [learners[i].pulled_mask() for i in range(m)]
    g = np.random.RandomState(seed + 555)
    ung, uno, unr = [], [], []
    for _ in range(80):
        for k in range(m):
            uns = np.where(~pulled[k])[0]
            if len(uns) >= cand:
                offu = g.choice(uns, size=cand, replace=False)
                ung.append(R[k, offu[int(np.argmax(preds[k][offu]))]]); uno.append(R[k, offu].max()); unr.append(R[k, offu].mean())
    gm, om, rm = np.mean(ung), np.mean(uno), np.mean(unr)
    unseen = float((gm - rm) / max(om - rm, 1e-6))
    cr, co, cd = np.cumsum(real), np.cumsum(orac), np.cumsum(rnd)
    anytime = float((cr[-1] - cd[-1]) / max(co[-1] - cd[-1], 1e-9))
    uniq = _state_uniq(learners, m)
    return name, mode, rho, seed, T, unseen, anytime, uniq


def main():
    print("=" * 96)
    print("E12 MASKING MODEL: persistent vs IID. m=%d n=%d d_hat=%d T=%d sigma_obs=%.2f %d seeds"
          % (pc.M, pc.N, pc.D_HAT, pc.T, pc.SB, len(SEEDS)))
    print("=" * 96)

    # Part A: unseen/anytime/uniq vs rho, both modes (T = default)
    jobsA = [(nm, mode, rho, s, pc.T) for mode in MODES for rho in RHOS for nm in METHODS for s in SEEDS]
    rawA = {mode: {str(rho): {nm: {"unseen": [], "anytime": [], "uniq": []} for nm in METHODS}
                   for rho in RHOS} for mode in MODES}
    # Part B: uniqueness vs T (RewardCF, both modes, rho=0.25)
    jobsB = [("RewardCF", mode, 0.25, s, T) for mode in MODES for T in TGRID for s in SEEDS]
    rawB = {mode: {str(T): {"uniq": [], "unseen": []} for T in TGRID} for mode in MODES}

    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_iid, j): ("A", j) for j in jobsA}
        futs.update({ex.submit(run_iid, j): ("B", j) for j in jobsB})
        done = 0; total = len(jobsA) + len(jobsB)
        for fut in as_completed(futs):
            tag, _ = futs[fut]
            nm, mode, rho, s, T, un, an, uq = fut.result()
            if tag == "A":
                c = rawA[mode][str(rho)][nm]; c["unseen"].append(un); c["anytime"].append(an); c["uniq"].append(uq)
            else:
                c = rawB[mode][str(T)]; c["uniq"].append(uq); c["unseen"].append(un)
            done += 1
            if done % 40 == 0 or done == total:
                print("  ... %d/%d cells done" % (done, total))

    for metric in ["unseen", "anytime"]:
        print("\n[Part A] %s skill: persistent vs iid (mean/%d seeds)" % (metric, len(SEEDS)))
        print("%9s | " % "rho" + " ".join("%14.2f" % r for r in RHOS))
        for nm in METHODS:
            line = "%9s | " % nm
            for rho in RHOS:
                p = np.mean(rawA["persistent"][str(rho)][nm][metric])
                i = np.mean(rawA["iid"][str(rho)][nm][metric])
                line += "%6.2f/%-6.2f " % (p, i)
            print(line + "  (persist/iid)")
    print("\n[Part A] state-UNIQUENESS (RewardCF): persistent vs iid")
    print("%9s | " % "rho" + " ".join("%14.2f" % r for r in RHOS))
    line = "%9s | " % "RewardCF"
    for rho in RHOS:
        p = np.nanmean(rawA["persistent"][str(rho)]["RewardCF"]["uniq"])
        i = np.nanmean(rawA["iid"][str(rho)]["RewardCF"]["uniq"])
        line += "%6.2f/%-6.2f " % (p, i)
    print(line + "  (persist/iid)")
    print("\n[Part B] state-UNIQUENESS vs T (RewardCF, rho=0.25): persistent vs iid")
    print("%9s | " % "T" + " ".join("%14d" % T for T in TGRID))
    line = "%9s | " % "uniq"
    for T in TGRID:
        p = np.nanmean(rawB["persistent"][str(T)]["uniq"])
        i = np.nanmean(rawB["iid"][str(T)]["uniq"])
        line += "%6.2f/%-6.2f " % (p, i)
    print(line + "  (persist/iid)")
    print("\nTheorem 4: unseen/anytime ~invariant to mode; uniq durable (persistent) vs transient")
    print("(iid lower, and DECREASING in T).")

    path = save_results("e12_iid_masking", {
        "meta": {"experiment": "E12 persistent vs iid masking (Theorem 4)",
                 "methods": METHODS, "modes": MODES, "rhos": RHOS, "seeds": SEEDS,
                 "Tgrid": TGRID, "m": pc.M, "n": pc.N, "d": pc.D, "K": pc.K,
                 "d_hat": pc.D_HAT, "T": pc.T, "sigma_obs": pc.SB, "sigma_own": pc.SO,
                 "metric": "unseen+anytime+state-uniqueness; PartA vs rho, PartB uniq vs T"},
        "rawA": rawA, "rawB": rawB})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
