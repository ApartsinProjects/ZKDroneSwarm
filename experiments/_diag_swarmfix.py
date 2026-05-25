"""Fast root-cause test: which LEGITIMATE change closes the SwarmCF-vs-kNN gap on the EMERGENT
physical sensing reward (S=4)? Tests (a) biases, (b) lower ridge, (c) factor+neighborhood hybrid.
Run:  python experiments/_diag_swarmfix.py
"""
import os, sys, time, copy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentswarm.sweeps import base_config
from emergent_lowrank import physical_reward, wide_factors, run_one

SEEDS = list(range(8))
S = 4
t0 = time.time()
base = base_config(rho=0.25)

def cfg_with(**kw):
    c = copy.deepcopy(base)
    for k, v in kw.items():
        setattr(c, k, v)
    return c

# (label, algo, cfg) triples
runs = [
    ("kNN-CF (ref)",        "knn_cf",        base),
    ("SwarmCF (base)",      "swarm_cf",      base),
    ("SwarmCF ridge=0.3",   "swarm_cf",      cfg_with(ridge=0.3)),
    ("SwarmCF ridge=0.1",   "swarm_cf",      cfg_with(ridge=0.1)),
    ("SwarmCF+bias",        "swarm_cf_bias", base),
    ("SwarmCF+bias r=0.3",  "swarm_cf_bias", cfg_with(ridge=0.3)),
    ("SwarmCF+nbr",         "swarm_cf_nbr",  base),
    ("SwarmCF+nbr r=0.3",   "swarm_cf_nbr",  cfg_with(ridge=0.3)),
]

res = {lab: [] for lab, _, _ in runs}
for s in SEEDS:
    wr = np.random.RandomState(s)
    R = physical_reward(wr, base.m, base.n, S)
    P, U = wide_factors(R)
    d_guess = base.rank_for_run(wr)            # same guessed rank for every method this seed
    for lab, algo, c in runs:
        res[lab].append(run_one(c, P, U, d_guess, s, algo))
    print("  seed %d done (%.0fs)" % (s, time.time() - t0), flush=True)

print("\n=== SwarmCF-fix test on emergent physical reward (S=4, rho=0.25, %d seeds) ===" % len(SEEDS))
for lab, _, _ in runs:
    a = np.array(res[lab], float)
    print("  %-22s %.3f +/- %.3f" % (lab, a.mean(), a.std() / np.sqrt(len(a))))
print("(%.0fs)" % (time.time() - t0))
