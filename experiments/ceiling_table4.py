"""Table 4 / Section 6.4 ceiling: how much of the centralized full-communication Hungarian-matching
ceiling does communication-free SwarmCF recover, on the uniform world?

run_contention_cell returns anytime = (earned - random) / (Hungarian_match - random), i.e. the
fraction of the centralized capacity-1 matching ceiling (ceiling = 1.0, random = 0.0). We report it
for SwarmCF and the structure-free floor at the all-tasks pool (|S| = n), masked rho = 0.25, 16 seeds.
Single process (no pool). Run from repo root: python experiments/ceiling_table4.py
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
from latentswarm.sweeps import base_config, _cfg_for, run_contention_cell

cfg = _cfg_for(base_config(), rho=0.25)          # uniform_cosine, m=30, n=240, T=50, 16 seeds
POOL = cfg.n                                       # all-tasks menu (|S| = n)
SEEDS = list(cfg.seeds)


def boot(a, B=10000, seed=0):
    a = np.asarray(a, float); rng = np.random.RandomState(seed)
    bs = a[rng.randint(0, len(a), (B, len(a)))].mean(1)
    return a.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


print("Table 4 ceiling (uniform world, rho=0.25, all-tasks pool |S|=%d, capacity-1, %d seeds)" % (POOL, len(SEEDS)))
print("fraction of the centralized Hungarian-matching ceiling (ceiling=1.0, random=0.0):")
for algo in ["swarm_cf", "ucb_indep"]:
    xs = [run_contention_cell(cfg, algo, POOL, s)[0] for s in SEEDS]
    m, lo, hi = boot(xs)
    print("  %-10s %.3f [%.3f, %.3f]" % (algo, m, lo, hi))
