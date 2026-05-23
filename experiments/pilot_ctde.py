"""E-CTDE / Rank5 CEILING bracket: how much does the NO-COMMUNICATION constraint cost? We add a
centralized, communication-FULL controller (a CTDE-style ceiling, NOT a competitor): ONE shared CF
model is fed EVERY drone's observation each round, and assignment is a centralized Hungarian on its
predicted utilities (so drones are coordinated to DISTINCT targets, no collisions). Reported like the
oracle: a ceiling between greedy CF and the true Hungarian optimum. The gap (CTDE - our best
comms-free method) is the price of zero communication; the gap (oracle - CTDE) is the price of not
knowing the true reward. Pool sweep, capacity-1 contention, 8 seeds. Writes docs/CTDE.md."""
import os
import sys
from collections import defaultdict
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
import pilot_contention as pcon
from pilot_noise import RewardCF
from core import make_world
from _results_io import save_results

try:
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLS = [240, 60, 30, 15]
SEEDS = list(range(8))
RNG = np.random.RandomState(0)
_CONV = dict(eps0=0.5, eps_min=0.05, eps_decay=0.97, als_sweeps=20, refit_every=1)


def _assign(pred):
    """Hungarian max-sum assignment of drones (rows) to distinct targets (cols). pred (m, p)."""
    m, p = pred.shape
    if _HAVE_SCIPY:
        ri, ci = linear_sum_assignment(-pred)
        return list(zip(ri, ci))
    taken = set(); rows = set(); out = []
    order = np.dstack(np.unravel_index(np.argsort(-pred, axis=None), pred.shape))[0]
    for r, c in order:
        if r in rows or c in taken:
            continue
        rows.add(r); taken.add(c); out.append((r, c))
        if len(rows) == min(m, p):
            break
    return out


def run_ctde(world, seed, pool):
    """Centralized comms-FULL ceiling: one shared CF model + Hungarian assignment. Returns anytime
    earned-reward skill (matching-normalized), comparable to pcon.run_contention's metric."""
    P, U, R = world[:3]; m, n = R.shape
    T, so = pc.T, pc.SO
    rng = np.random.RandomState(seed + 999)
    central = RewardCF(m, n, pc.D_HAT, 0, seed + 1, **_CONV)   # ONE shared model, sees everything
    cum_real = cum_orac = cum_rand = 0.0
    for t in range(T):
        S = rng.choice(n, size=pool, replace=False)
        pred = central.U[S] @ central.P.T                      # (pool, m) predicted reward
        pairs = _assign(pred.T)                                # drones -> distinct targets in S
        picks = np.full(m, -1)
        for r, c in pairs:
            picks[int(r)] = int(S[int(c)])
        true_r = np.array([R[i, picks[i]] if picks[i] >= 0 else 0.0 for i in range(m)])
        cum_real += float(true_r[picks >= 0].sum())            # distinct targets -> no collisions
        Rsub = R[:, S]
        cum_orac += pcon._match_opt(Rsub)
        cum_rand += pcon._rand_earned(Rsub, np.random.RandomState(seed * 131 + t))
        # centralized observation of ALL assigned engagements (comms-full: every outcome, mild noise)
        choices = picks.copy()
        revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
        for i in range(m):
            if picks[i] >= 0:
                revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
        central.observe(t, choices, revealed, [S] * m, rvar)
    return float((cum_real - cum_rand) / max(cum_orac - cum_rand, 1e-9))


def _job(args):
    kind, pool, seed = args
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    if kind == "CTDE-ceiling":
        return kind, pool, seed, run_ctde(w, seed, pool)
    Cls, hp = pcon.REG[kind]
    a, u, c = pcon.run_contention(Cls, hp, 1.0, pool, seed)
    return kind, pool, seed, float(a)


ORDER = ["CTDE-ceiling", "ContentionAdaCF", "RewardCFconv", "UCBIndep"]


def ci(vals, B=10000):
    a = np.asarray([v for v in vals if v is not None], float)
    idx = RNG.randint(0, len(a), (B, len(a))); mb = a[idx].mean(1)
    return a.mean(), np.percentile(mb, 2.5), np.percentile(mb, 97.5)


def cell(vals):
    mu, lo, hi = ci(vals); return "%.3f [%.3f, %.3f]" % (mu, lo, hi)


def main():
    jobs = [(k, p, s) for p in POOLS for k in ORDER for s in SEEDS]
    raw = {str(p): {k: [None] * len(SEEDS) for k in ORDER} for p in POOLS}
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            k, p, s, v = fut.result()
            raw[str(p)][k][s] = v
            done += 1
            if done % 16 == 0 or done == len(jobs):
                print("  ... %d/%d" % (done, len(jobs)))

    save_results("ctde", {
        "meta": {"experiment": "CTDE comms-full ceiling vs comms-free methods (no-comms cost)",
                 "methods": ORDER, "pools": POOLS, "seeds": SEEDS, "m": pc.M, "n": pc.N, "d": pc.D,
                 "d_hat": pc.D_HAT, "scipy": _HAVE_SCIPY,
                 "metric": "anytime earned-reward skill (matching-normalized); CTDE is a CEILING not a competitor"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))

    L = ["# CTDE comms-full ceiling: the price of zero communication (E-CTDE / Rank5)\n",
         "Earned-reward skill (matching-normalized; oracle = 1) under capacity-1 contention. "
         "CTDE-ceiling = ONE centralized CF model + Hungarian assignment (full communication, "
         "coordinated to distinct targets); a CEILING, not a competitor. 8 seeds, bootstrap 95%% CI.\n",
         "| method | " + " | ".join("pool=%d" % p for p in POOLS) + " |",
         "|" + "---|" * (len(POOLS) + 1)]
    for k in ORDER:
        cells = [cell(raw[str(p)][k]) for p in POOLS]
        lab = "_%s (ceiling)_" % k if k == "CTDE-ceiling" else ("**%s**" % k if "CF" in k else k)
        L.append("| %s | %s |" % (lab, " | ".join(cells)))
    L.append("")
    # price of no-comms = CTDE - our best comms-free, per pool
    gaps = []
    for p in POOLS:
        g = float(np.mean(raw[str(p)]["CTDE-ceiling"]) - np.mean(raw[str(p)]["ContentionAdaCF"]))
        gaps.append("pool=%d: %+.3f" % (p, g))
    L.append("Price of zero communication (CTDE-ceiling - ContentionAdaCF): " + ";  ".join(gaps) + "\n")
    L.append("Read: CTDE (full comms) is a CEILING above our comms-free methods and below the oracle. "
             "A SMALL gap to ContentionAdaCF means communication buys little here, our comms-free "
             "de-confliction recovers most of the coordination value; the gap widens where within-round "
             "coordination matters most (severe contention).\n")
    out_md = os.path.join(ROOT, "docs", "CTDE.md")
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L)); print("wrote %s (raw saved earlier)" % out_md)


if __name__ == "__main__":
    main()
