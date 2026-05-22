"""
C8 (GROUNDBREAKING CANDIDATE): generalization to UNSEEN agent-task pairs.

The categorical edge of matrix decomposition: it predicts the reward of pairs an
agent has NEVER observed (from the latent factors). Independent/tabular learning
is at the FLOOR by construction on unseen pairs (no estimate -> prior/mean).

Setting (in-scope): swarm, unknown low-rank structure, limited observability,
no communication (public broadcast of actions+rewards), honest agents, online.

Metric M1 -- UNSEEN-PAIR skill: after training, restrict each drone to targets
it NEVER pulled; offer random subsets of ONLY-unseen targets; greedy-pick with
the final model. skill=(reward-rand)/(oracle-rand) over unseen. Tabular ~ 0 by
construction; CF (RewardCF/BothCF) >> 0 = categorical win.

Also reports OVERALL skill (seen+unseen) for reference.
"""
import numpy as np
import sys
sys.stdout.reconfigure(encoding="utf-8")

from pilot_noise import Tabular, RewardCF, BothCF, make_world_struct
from _results_io import save_results


def train_then_eval(Cls, hp, world, T, p, seed, so, sb, cand, d_hat):
    P, U, R = world; m, n = R.shape
    rng = np.random.RandomState(seed + 999)
    # FAIRNESS: learners use a FIXED guessed rank d_hat (NOT the true rank).
    learners = [Cls(m, n, d_hat, i, seed + 7 * i + 1, **hp) for i in range(m)]
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i and rng.rand() < p:
                    revealed[k] = true_r[k] + rng.normal(0, sb); rvar[k] = sb ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    preds = [learners[i].predict_scores() for i in range(m)]
    pulled = [learners[i].pulled_mask() for i in range(m)]
    g = np.random.RandomState(seed + 555)
    ov_got, ov_oc, ov_rd = [], [], []        # overall (any target)
    un_got, un_oc, un_rd = [], [], []        # unseen targets only
    for _ in range(120):
        for k in range(m):
            off = g.choice(n, size=cand, replace=False)
            ov_got.append(R[k, off[int(np.argmax(preds[k][off]))]])
            ov_oc.append(R[k, off].max()); ov_rd.append(R[k, off].mean())
            unseen = np.where(~pulled[k])[0]
            if len(unseen) >= cand:
                offu = g.choice(unseen, size=cand, replace=False)
                un_got.append(R[k, offu[int(np.argmax(preds[k][offu]))]])
                un_oc.append(R[k, offu].max()); un_rd.append(R[k, offu].mean())

    def sk(got, oc, rd):
        gm, om, rm = np.mean(got), np.mean(oc), np.mean(rd)
        return (gm - rm) / max(om - rm, 1e-6)
    return sk(ov_got, ov_oc, ov_rd), sk(un_got, un_oc, un_rd)


def run(Cls, hp, cfg, m, n, d, T, cand, p, so, sb, seeds, d_hat):
    ov, un = [], []
    for s in seeds:
        w = make_world_struct(m, n, d, cfg["structure"], cfg["eps"], cfg["n_clusters"], cfg["sharp"], s)
        o, u = train_then_eval(Cls, hp, w, T, p, s, so, sb, cand, d_hat)
        ov.append(o); un.append(u)
    return ov, un


def main():
    m, n, d, T, cand = 30, 240, 5, 50, 20
    d_hat = 8   # FAIRNESS: CF uses a fixed GUESSED rank (true d=5); not oracle.
    so = sb = 0.10
    seeds = list(range(5))
    base = dict(eps0=0.5, eps_min=0.05, eps_decay=0.93, als_sweeps=10, refit_every=2)
    methods = {
        "Tabular":  (Tabular,  dict(eps0=0.5, eps_min=0.05, eps_decay=0.93)),
        "RewardCF": (RewardCF, dict(**base)),
        "BothCF":   (BothCF,   dict(comp=True, s2c=0.2, n_neg=1, within=True, warm_frac=0.3, T_total=T, **base)),
    }
    print("=" * 80)
    print("C8: GENERALIZATION TO UNSEEN PAIRS. m=%d n=%d d=%d T=%d cand=%d %d seeds, clustG nc15"
          % (m, n, d, T, cand, len(seeds)))
    print("each drone pulls ~%d/%d distinct targets in training; UNSEEN = the rest." % (T, n))
    print("skill=(greedy-rand)/(oracle-rand). UNSEEN: tabular ~0 by construction.")
    print("=" * 80)
    print(f"{'method':10s} | {'OVERALL skill':>16s} | {'UNSEEN-PAIR skill':>18s}")
    print("-" * 54)
    cfg = dict(structure="cluster_gauss", eps=0.10, n_clusters=15, sharp=1.0)
    print(f"(FAIR: CF uses guessed rank d_hat={d_hat}, true d={d})")
    raw = {}
    for name, (Cls, hp) in methods.items():
        ov, un = run(Cls, hp, cfg, m, n, d, T, cand, 1.0, so, sb, seeds, d_hat)
        raw[name] = {"overall": list(ov), "unseen": list(un)}
        print(f"{name:10s} | {np.mean(ov):7.3f} +/- {np.std(ov):5.3f} | "
              f"{np.mean(un):8.3f} +/- {np.std(un):5.3f}")
    print("-" * 54)
    path = save_results("c8_generalize", {
        "meta": {"experiment": "C8 unseen-pair generalization", "m": m, "n": n,
                 "d": d, "d_hat": d_hat, "T": T, "cand": cand, "seeds": list(seeds), "p_share": 1.0,
                 "sigma_own": so, "sigma_obs": sb, "structure": cfg,
                 "metrics": "per-seed overall skill and unseen-pair skill"},
        "raw": raw})
    print("complete data saved ->", path)
    print("CATEGORICAL WIN if CF UNSEEN-PAIR skill >> Tabular's (~0). Tabular cannot")
    print("predict pairs it never observed; CF generalizes via the latent factors.")


if __name__ == "__main__":
    main()
