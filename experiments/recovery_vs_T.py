"""Recovery versus horizon: does the actual eps-greedy policy enter the Theorem 2 regime?

Sweeps the mission horizon T from the body's T=50 across the worst-case full-recovery time
T_rec = O((n d / rho m) log n) (numerically ~880 rounds at the headline n=240, d=5, m=30, rho=0.25),
and reports, for the ACTUAL eps-decay SwarmCF policy:
  (a) held-out unseen-pair skill, and
  (b) the Theorem 2 RECOVERY FRACTION: the share of (robot i, task j) pairs whose spanning condition
      holds given the actual coverage (j engaged by i, OR p_i in span{p_k : k a visible engager of j}).
alongside the recovery fraction of a NON-ADAPTIVE uniform-exploration policy (random), the regime
Theorem 2 assumes.

Story: uniform exploration reaches full recovery near T_rec (validating the theorem's regime), while
the eps-greedy policy attains high unseen skill far earlier through partial/graded recovery of the
high-value tasks (why the body operates at T=50), its full recovery being slower (consistent with the
eps-floor corollary's 1/eps_min inflation, since exploitation starves low-value tasks of coverage).

Run from repo root:  python experiments/recovery_vs_T.py
"""
import os
import sys
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentswarm.sweeps import base_config, _cfg_for
from latentswarm.registry import ALGORITHMS, get
from latentswarm.scenarios import build_scenario
from latentswarm.env import ZKMRTAEnv
from latentswarm.metrics import UnseenPairSkill, bootstrap_ci

TS = [50, 100, 200, 400, 800]
SEEDS = list(range(16))
RHO = 0.25
OUT_PNG = os.path.join(ROOT, "docs", "figures", "F_recoveryT.png")
OUT_PDF = os.path.join(ROOT, "docs", "figures", "pdf", "F_recoveryT.pdf")


def recovery_fraction(P, n, mask, engaged):
    """Theorem 2 recovery fraction: share of (i, j) pairs predictable given the realized coverage.
    A pair (i, j) counts as recovered if i engaged j, or p_i lies in the span of the factors of the
    teammates visible to i that engaged j (the spanning condition)."""
    m, d = P.shape
    eng = [set(int(j) for j in e) for e in engaged]
    engagers = [[] for _ in range(n)]
    for k in range(m):
        for j in eng[k]:
            engagers[j].append(k)
    rec = 0
    for i in range(m):
        pi = P[i]; npi = np.linalg.norm(pi) + 1e-12
        for j in range(n):
            if j in eng[i]:
                rec += 1
                continue
            E = [k for k in engagers[j] if mask[i, k]]
            if not E:
                continue
            B = P[E]                                  # (|E|, d): rows are the engager factors p_k
            c, *_ = np.linalg.lstsq(B.T, pi, rcond=None)   # solve B^T c = p_i (p_i in span of rows?)
            if np.linalg.norm(B.T @ c - pi) <= 1e-6 * npi:
                rec += 1
    return rec / (m * n)


def run_one(cfg, P, U, d_guess, seed, algo):
    env = ZKMRTAEnv(cfg, P, U, d_guess, seed=seed)
    pol = get(ALGORITHMS, algo)(cfg, cfg.m, cfg.n, d_guess, seed=1000 + seed)
    obs = env.reset()
    for _ in range(cfg.T):
        a = pol.act(obs)
        obs, _r, _info = env.step(a)
        pol.observe(obs)
    pred = pol.predict_rows()
    useed = np.random.RandomState(10000 + seed)
    skill = UnseenPairSkill().compute(P=P, U=U, pred_rows=pred, engaged=env.engaged, rng=useed)
    recov = recovery_fraction(P, cfg.n, env.mask, env.engaged)
    return (float(skill) if skill is not None else None), float(recov)


def main():
    base = base_config(rho=RHO)
    T_rec = (base.n * base.d / (RHO * base.m)) * np.log(base.n)   # numeric Theorem 2 estimate (const 1)
    t0 = time.time()
    res = {str(T): {"swarm_cf": {"skill": [], "recov": []}, "random": {"recov": []}} for T in TS}
    print("[recovery_vs_T] T_rec ~ %.0f rounds; sweeping T=%s, %d seeds" % (T_rec, TS, len(SEEDS)), flush=True)
    for T in TS:
        cfg = _cfg_for(base, T=T)
        for s in SEEDS:
            wr = np.random.RandomState(s)
            P, U = build_scenario(cfg, wr).generate()
            d_guess = cfg.rank_for_run(wr)
            sk, rc = run_one(cfg, P, U, d_guess, s, "swarm_cf")
            res[str(T)]["swarm_cf"]["skill"].append(sk)
            res[str(T)]["swarm_cf"]["recov"].append(rc)
            _, rcu = run_one(cfg, P, U, d_guess, s, "random")
            res[str(T)]["random"]["recov"].append(rcu)
        print("[recovery_vs_T] T=%-4d done (%.0fs)" % (T, time.time() - t0), flush=True)

    # aggregate
    def agg(key, sub):
        return [bootstrap_ci(res[str(T)][key][sub]) for T in TS]
    sk = agg("swarm_cf", "skill"); rc = agg("swarm_cf", "recov"); rcu = agg("random", "recov")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    xs = np.array(TS)
    ax[0].errorbar(xs, [a[0] for a in sk], yerr=[[a[0] - a[1] for a in sk], [a[2] - a[0] for a in sk]],
                   marker="o", color="C0", capsize=3, label="SwarmCF unseen-pair skill")
    ax[0].axvline(T_rec, color="gray", ls="--", lw=1.2, label="$T_{\\mathrm{rec}}$ (Theorem 2)")
    ax[0].axvline(50, color="C2", ls=":", lw=1.2, label="body $T=50$")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("horizon $T$ (rounds)"); ax[0].set_ylabel("unseen-pair skill")
    ax[0].set_title("(a) Operational skill climbs with the horizon", fontsize=10)
    ax[0].grid(alpha=0.25); ax[0].legend(fontsize=8, loc="lower right")
    ax[1].errorbar(xs, [a[0] for a in rcu], yerr=[[a[0] - a[1] for a in rcu], [a[2] - a[0] for a in rcu]],
                   marker="s", color="C3", capsize=3, label="uniform exploration (Theorem 2 regime)")
    ax[1].errorbar(xs, [a[0] for a in rc], yerr=[[a[0] - a[1] for a in rc], [a[2] - a[0] for a in rc]],
                   marker="o", color="C0", capsize=3, label="$\\varepsilon$-greedy (the policy we run)")
    ax[1].axvline(T_rec, color="gray", ls="--", lw=1.2, label="$T_{\\mathrm{rec}}$")
    ax[1].axhline(1.0, color="k", lw=0.6, ls=":")
    ax[1].set_xlabel("horizon $T$ (rounds)"); ax[1].set_ylabel("recovery fraction (spanning condition)")
    ax[1].set_title("(b) Recovery vs horizon: uniform reaches the regime near $T_{rec}$", fontsize=10)
    ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.25); ax[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close()

    out_path = os.path.join(ROOT, "results", "pilots", "recovery_vs_T_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    json.dump({"meta": {"experiment": "recovery vs horizon", "T_rec": T_rec, "Ts": TS,
                        "seeds": SEEDS, "rho": RHO, "eps_min": base.epsilon_min},
               "results": res}, open(out_path, "w"), indent=1)

    print("\n=== recovery vs horizon (rho=%.2f, %d seeds; T_rec~%.0f) ===" % (RHO, len(SEEDS), T_rec))
    print("%-6s | %-22s | %-22s | %-22s" % ("T", "SwarmCF unseen skill", "SwarmCF recov frac", "uniform recov frac"))
    for i, T in enumerate(TS):
        print("%-6d | %.3f [%.3f,%.3f]    | %.3f [%.3f,%.3f]    | %.3f [%.3f,%.3f]"
              % (T, sk[i][0], sk[i][1], sk[i][2], rc[i][0], rc[i][1], rc[i][2], rcu[i][0], rcu[i][1], rcu[i][2]))
    print("\nsaved -> %s  and  %s  (%.0fs)" % (OUT_PNG, out_path, time.time() - t0))


if __name__ == "__main__":
    main()
