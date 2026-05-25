"""Does more OBSERVATION noise favour the low-rank learner over memory-based kNN-CF? Low-rank
factorization pools all observations to estimate few parameters (a denoiser), and its prediction
P_i[i]*U_j leans on robot i's OWN near-clean factor (sigma_own) times a pooled/denoised task factor;
kNN-CF instead averages teammates' RAW noisy readings AND computes similarities from noisy
co-observations, so it is doubly exposed. Prediction: the kNN-over-SwarmCF gap shrinks (and reverses
with the bias fix) as sigma_obs grows, until extreme noise sends everyone to the floor.
Run:  python experiments/_diag_noise.py
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentswarm.sweeps import base_config
from emergent_lowrank import physical_reward, wide_factors, run_one

METHODS = ["knn_cf", "swarm_cf", "swarm_cf_bias", "swarm_cf_nbr"]
PRETTY = {"knn_cf": "kNN-CF", "swarm_cf": "SwarmCF", "swarm_cf_bias": "SwarmCF+bias", "swarm_cf_nbr": "SwarmCF+nbr"}


def eval_noise(seeds, sigma_obs, S=4, range_frac=2.0, rho=0.25, sigma_own=0.10):
    cfg = base_config(rho=rho); cfg.sigma_obs = float(sigma_obs); cfg.sigma_own = float(sigma_own)
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
    seeds = list(range(8))
    grid = [0.1, 0.3, 0.5, 0.7, 1.0]
    print("=== Observation-noise (sigma_obs) sweep (S=4, rho=0.25, sigma_own=0.10; %d seeds) ===" % len(seeds))
    print("sig_obs | " + " ".join("%-12s" % PRETTY[m] for m in METHODS) + " | gap(bias-kNN)")
    for sg in grid:
        r = eval_noise(seeds, sg)
        mu = {a: float(np.mean(r[a])) for a in METHODS}
        print("%.2f    | " % sg + " ".join("%-12.3f" % mu[a] for a in METHODS)
              + " | %+.3f" % (mu["swarm_cf_bias"] - mu["knn_cf"]), flush=True)
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
