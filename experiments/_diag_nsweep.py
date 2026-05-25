"""Does increasing the number of TARGETS n (task-scarcity, the paper's regime) favor the global
low-rank learner over the memory-based neighborhood learner? With m, T fixed, observations-per-task
~ m*T/n, so larger n = sparser per task. Prediction: kNN-CF (needs a same-task neighbor) degrades
faster than SwarmCF+bias (borrows strength across the shared d-dim structure; 1-2 obs locate a task).
Run:  python experiments/_diag_nsweep.py
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentswarm.sweeps import base_config
from emergent_lowrank import physical_reward, wide_factors, run_one

METHODS = ["knn_cf", "swarm_cf", "swarm_cf_bias", "swarm_cf_nbr"]
PRETTY = {"knn_cf": "kNN-CF", "swarm_cf": "SwarmCF", "swarm_cf_bias": "SwarmCF+bias", "swarm_cf_nbr": "SwarmCF+nbr"}


def eval_n(seeds, N, S=4, range_frac=2.0, rho=0.25):
    cfg = base_config(rho=rho); cfg.n = int(N)
    out = {a: [] for a in METHODS}
    for s in seeds:
        wr = np.random.RandomState(s)
        R = physical_reward(wr, cfg.m, cfg.n, S, range_frac=range_frac)
        P, U = wide_factors(R)
        d_guess = cfg.rank_for_run(wr)
        for a in METHODS:
            out[a].append(run_one(cfg, P, U, d_guess, s, a))
    return out


def main():
    t0 = time.time()
    seeds = list(range(6))
    n_grid = [120, 240, 480, 720]
    sweep = {}
    for N in n_grid:
        sweep[N] = eval_n(seeds, N)
        print("  n=%d done (%.0fs)" % (N, time.time() - t0), flush=True)
    print("\n=== TARGET-COUNT sweep (m=30, T=50, S=4, range_frac=2.0, rho=0.25; 6 seeds) ===")
    print("  obs/task ~ m*T/n:  " + "  ".join("n=%d:%.1f" % (N, 30 * 50 / N) for N in n_grid))
    print("  %-14s " % "method" + " ".join("n=%-5d" % N for N in n_grid))
    for a in METHODS:
        mu = [float(np.mean(sweep[N][a])) for N in n_grid]
        print("  %-14s " % PRETTY[a] + " ".join("%.3f" % v for v in mu))
    print("  %-14s " % "gap(bias-kNN)" + " ".join(
        "%+.3f" % (float(np.mean(sweep[N]["swarm_cf_bias"])) - float(np.mean(sweep[N]["knn_cf"]))) for N in n_grid))
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
