"""
C16 (ANYTIME / AUC): the OPERATIONALLY relevant metric -- cumulative reward
actually EARNED during the T rounds ("targets destroyed by round K"), not just the
quality of the FINAL policy. This CHARGES the cost of any probe/explore phase that
the cycle-15 final-policy "skill" metric gave away for free.

Why this is the fair anytime test (and why CF should separate cleanly):
  - n=240 targets but only T=50 rounds => a per-(drone,target) bandit (UCBIndep)
    cannot pull each arm even once; its offer almost always contains an UNTRIED
    target (infinite UCB bonus), so it explores FOREVER and earns ~random every
    round. No generalization => no anytime exploitation in the sample-starved
    regime.
  - Explore-then-commit low-rank methods (ESTR, PTF) earn ~random during their
    40%-of-rounds probe phase, then jump after the fit.
  - Our ONLINE weighted-ALS (RewardCF/BothCF) learns the low-rank structure as it
    goes and exploits UNSEEN targets immediately => rises from the first rounds.

Metric: at each round each drone is offered `cand` random targets and earns the
TRUE reward of its pick. Per round, normalize against oracle (best-in-offer) and
random (mean-in-offer); report the CUMULATIVE normalized skill trajectory
traj[t] = (sum_{<=t} realized - sum_{<=t} random) / (sum_{<=t} oracle - random).
traj[-1] = whole-episode anytime skill (charges probe cost fully). Saves full
per-seed trajectories for a figure (F6). Reuses pilot_compare config + registry.
"""
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.stdout.reconfigure(encoding="utf-8")

import pilot_compare as pc
from core import make_world
from _results_io import save_results

RHOS = [1.0, 0.25]
SEEDS = list(range(8))


def run_anytime_clshp(Cls, hp, rho, seed, d_hat=None):
    """Core anytime loop driven by an explicit (Cls, hp) -- so ablation variants not
    in pc.REGISTRY can be evaluated through the SAME loop. Returns the cumulative-
    normalized skill trajectory traj[T] (traj[-1] = whole-episode anytime skill)."""
    d_hat = pc.D_HAT if d_hat is None else d_hat
    w = make_world(pc.M, pc.N, pc.D, pc.K, pc.K, within=0.15, seed=seed, signed=True)
    P, U, R = w[:3]; m, n = R.shape
    T, cand, so, sb = pc.T, pc.CAND, pc.SO, pc.SB
    rng = np.random.RandomState(seed + 999)
    Mask = rng.rand(m, m) < rho; np.fill_diagonal(Mask, True)
    learners = [Cls(m, n, d_hat, i, seed + 7 * i + 1, **hp) for i in range(m)]
    real = np.zeros(T); orac = np.zeros(T); rnd = np.zeros(T)
    for t in range(T):
        cand_sets = [rng.choice(n, size=cand, replace=False) for _ in range(m)]
        choices = np.array([learners[i].select(t, cand_sets[i]) for i in range(m)])
        true_r = np.array([R[i, choices[i]] for i in range(m)])
        real[t] = float(true_r.sum())
        orac[t] = float(sum(R[i, cand_sets[i]].max() for i in range(m)))
        rnd[t] = float(sum(R[i, cand_sets[i]].mean() for i in range(m)))
        for i in range(m):
            revealed = np.full(m, np.nan); rvar = np.full(m, np.inf)
            revealed[i] = true_r[i] + rng.normal(0, so); rvar[i] = so ** 2
            for k in range(m):
                if k != i and Mask[i, k]:
                    revealed[k] = true_r[k] + rng.normal(0, sb); rvar[k] = sb ** 2
            learners[i].observe(t, choices, revealed, cand_sets, rvar)
    cr = np.cumsum(real); co = np.cumsum(orac); cd = np.cumsum(rnd)
    traj = (cr - cd) / np.maximum(co - cd, 1e-9)
    return traj.tolist()


def run_anytime(args):
    """One (method, rho, seed) -> (name, rho, seed, cumulative-normalized traj[T])."""
    name, rho, seed = args
    Cls, hp = pc.REGISTRY[name]
    return name, rho, seed, run_anytime_clshp(Cls, hp, rho, seed)


def main():
    print("=" * 96)
    print("C16 ANYTIME (cumulative-reward AUC): m=%d n=%d d=%d d_hat=%d T=%d cand=%d %d seeds"
          % (pc.M, pc.N, pc.D, pc.D_HAT, pc.T, pc.CAND, len(SEEDS)))
    print("n>>T (sample-starved): per-arm bandits can't pull every arm once. "
          "Metric charges probe cost. sigma_own=%.2f sigma_obs=%.2f" % (pc.SO, pc.SB))
    print("=" * 96)

    jobs = [(nm, r, s) for r in RHOS for nm in pc.ORDER for s in SEEDS]
    raw = {str(r): {nm: [] for nm in pc.ORDER} for r in RHOS}   # raw[rho][name] = [traj per seed]
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(run_anytime, j): j for j in jobs}
        done = 0
        for fut in as_completed(futs):
            nm, r, s, traj = fut.result()
            raw[str(r)][nm].append(traj)
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print("  ... %d/%d cells done" % (done, len(jobs)))

    T = pc.T
    for r in RHOS:
        print("\n" + "-" * 96)
        print("rho = %.2f  | anytime cumulative-normalized skill at rounds K=T/4,T/2,T "
              "(mean/%d seeds)" % (r, len(SEEDS)))
        print(f"{'method':>10s} {'group':>16s} | {'@K=%d' % (T // 4):>10s} "
              f"{'@K=%d' % (T // 2):>10s} {'@K=%d (final)' % T:>14s}")
        print("-" * 96)
        kq, kh = T // 4 - 1, T // 2 - 1
        for nm in pc.ORDER:
            A = np.array(raw[str(r)][nm])   # (seeds, T)
            mq, mh, mf = A[:, kq].mean(), A[:, kh].mean(), A[:, -1].mean()
            sf = A[:, -1].std()
            print(f"{nm:>10s} {pc.GROUP[nm]:>16s} | {mq:10.3f} {mh:10.3f} "
                  f"{mf:8.3f}+-{sf:4.3f}")
    print("-" * 96)
    print("READ: online CF (RewardCF/BothCF) earns from the FIRST rounds (anytime);")
    print("UCBIndep stuck ~low (n>>T: perpetual exploration); ESTR/PTF pay a probe phase.")

    path = save_results("c16_anytime", {
        "meta": {"experiment": "C16 anytime cumulative-reward AUC",
                 "methods": pc.ORDER, "group": pc.GROUP, "rhos": RHOS, "seeds": SEEDS,
                 "m": pc.M, "n": pc.N, "d": pc.D, "K": pc.K, "d_hat": pc.D_HAT,
                 "T": pc.T, "cand": pc.CAND, "sigma_own": pc.SO, "sigma_obs": pc.SB,
                 "metric": "per-seed cumulative-normalized skill trajectory traj[T]"},
        "raw": raw})
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
