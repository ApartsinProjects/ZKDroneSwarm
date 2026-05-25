"""Would the bias term help in Table 4? Compare plain SwarmCF vs bias-augmented SwarmCF on the body's
zero-mean signed-cosine rewards across the three Table-4 trait distributions (masked harness, rho=0.25),
with kNN-CF for reference. Bias should be a no-op where the reward is zero-mean (uniform, approx) and may
help where discrete types create per-task popularity (block).
Run:  python experiments/_diag_table4_bias.py
"""
import os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from latentswarm.sweeps import base_config, run_cell

SCEN = [("block", "block_cosine", {}),
        ("uniform", "uniform_cosine", {}),
        ("approx(eps=0.5)", "approx_lowrank", {"approx_eps": 0.5})]
ALGOS = ["swarm_cf", "swarm_cf_bias", "knn_cf"]
PRETTY = {"swarm_cf": "SwarmCF", "swarm_cf_bias": "SwarmCF+bias", "knn_cf": "kNN-CF"}


def mean_se(xs):
    a = np.array([x for x in xs if x is not None], float)
    return float(a.mean()), float(a.std() / np.sqrt(max(len(a), 1)))


def main():
    t0 = time.time()
    seeds = list(range(8))
    print("=== Bias term on Table-4 distributions (body masked harness, rho=0.25, %d seeds) ===" % len(seeds))
    print("%-16s %-16s %-16s %-16s | bias-plain" % ("scenario", "SwarmCF", "SwarmCF+bias", "kNN-CF"))
    for label, scen, extra in SCEN:
        cfg = base_config(rho=0.25, scenario=scen, **extra)
        res = {a: [] for a in ALGOS}
        for s in seeds:
            for a in ALGOS:
                res[a].append(run_cell(cfg, a, s)["unseen"])
        m = {a: mean_se(res[a]) for a in ALGOS}
        gap = m["swarm_cf_bias"][0] - m["swarm_cf"][0]
        print("%-16s %.3f+/-%.3f    %.3f+/-%.3f    %.3f+/-%.3f  | %+.3f"
              % (label, m["swarm_cf"][0], m["swarm_cf"][1], m["swarm_cf_bias"][0], m["swarm_cf_bias"][1],
                 m["knn_cf"][0], m["knn_cf"][1], gap), flush=True)
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
