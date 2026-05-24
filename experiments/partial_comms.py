"""Partial-communication reference point: decomposing the gap to the centralized ceiling.

Section 6.4 attributes the gap between communication-free SwarmCF and the centralized
full-communication Hungarian ceiling to two forces: imperfect estimation and within-round
coordination (capacity-1 collisions under the shared all-tasks menu). This experiment isolates the
coordination component by adding a MINIMAL communication budget.

We compare, under the all-tasks menu with capacity-1 contention (the worst case for collisions,
rho=0.5, 16 seeds), three points on the communication spectrum:

  communication-free SwarmCF   our method: no messages; loses to BOTH estimation error and collisions.
  comm-coordinated SwarmCF     comm_cf: identical estimator, plus O(m) coordination messages per round
                               (each robot announces its claim; sequential greedy over unclaimed tasks),
                               so collisions are removed by construction and only estimation error remains.
  centralized Hungarian ceiling = 1.00 by normalization (perfect estimation AND perfect coordination).

Earned skill is normalized so the centralized capacity-1 Hungarian matcher is 1.0 and the random
floor is 0.0 (the Table 4 metric). The three points therefore decompose the ceiling gap into a
coordination part (communication-free -> coordinated) and an estimation part (coordinated -> 1.0).

Run from repo root:  python experiments/partial_comms.py
"""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from latentswarm.config import RunConfig
from latentswarm.registry import ALGORITHMS, get
from latentswarm.scenarios import build_scenario
from latentswarm.env import ZKMRTAEnv
from latentswarm.metrics import hungarian_oracle_per_step, EarnedSkill, bootstrap_ci

# communication-free context (random, structure-free, two structure-sharing baselines), our
# communication-free method, and the partial-communication coordinated variant.
ALGOS = ["random", "ucb_indep", "club", "knn_cf", "swarm_cf", "comm_cf"]
SEEDS = list(range(16))
PRETTY = {"random": "Random", "ucb_indep": "Indep-UCB", "club": "CLUB", "knn_cf": "KNNCF",
          "swarm_cf": "SwarmCF (comm-free)", "comm_cf": "SwarmCF + coordination (O(m) msgs/round)"}


def run_one(cfg, P, U, d_guess, seed, algo):
    """One all-tasks-menu, capacity-1 mission; returns (mean per-round reward, mean collision rate)."""
    env = ZKMRTAEnv(cfg, P, U, d_guess, seed=seed)
    pol = get(ALGORITHMS, algo)(cfg, cfg.m, cfg.n, d_guess, seed=1000 + seed)
    obs = env.reset()
    rew, coll = [], []
    for _ in range(cfg.T):
        actions = pol.act(obs)
        obs, rewards, info = env.step(actions)
        pol.observe(obs)
        rew.append(float(np.mean(rewards)))
        coll.append(info["collisions"] / cfg.m)
    return float(np.mean(rew)), float(np.mean(coll))


def main():
    em = EarnedSkill()
    raw = {a: {"earned": [], "coll": []} for a in ALGOS}
    for s in SEEDS:
        wr = np.random.RandomState(s)                         # same world construction as latentswarm.run
        cfg = RunConfig(offer_size=0, seeds=[s], scenario="uniform_cosine")   # all tasks, capacity-1
        P, U = build_scenario(cfg, wr).generate()
        d_guess = cfg.rank_for_run(wr)
        oracle = hungarian_oracle_per_step(P, U)
        rnd_mean, _ = run_one(cfg, P, U, d_guess, s, "random")
        for a in ALGOS:
            er, cr = run_one(cfg, P, U, d_guess, s, a)
            raw[a]["earned"].append(em.compute(mean_reward=er, random_mean=rnd_mean, oracle_mean=oracle))
            raw[a]["coll"].append(cr)

    print("=" * 84)
    print("Partial-communication decomposition (all-tasks menu, capacity-1, rho=0.5, %d seeds)" % len(SEEDS))
    print("earned = fraction of the centralized capacity-1 Hungarian ceiling (1.0); coll = collision rate")
    print("=" * 84)
    print("%-40s | %-22s | %-18s" % ("policy", "earned (frac of ceiling)", "collision rate"))
    print("-" * 84)
    res = {}
    for a in ALGOS:
        e = bootstrap_ci(raw[a]["earned"]); c = bootstrap_ci(raw[a]["coll"])
        res[a] = {"earned": e, "coll": c}
        print("%-40s | %6.3f [%6.3f,%6.3f]   | %6.3f [%6.3f,%6.3f]"
              % (PRETTY[a], e[0], e[1], e[2], c[0], c[1], c[2]))
    print("%-40s | %6.3f                  | %6.3f" % ("Centralized Hungarian (ceiling)", 1.0, 0.0))

    sf = res["swarm_cf"]["earned"][0]; cc = res["comm_cf"]["earned"][0]
    print("\nDecomposition of the gap to the ceiling (1.000):")
    print("  communication-free SwarmCF earns        %.3f" % sf)
    print("  + O(m)-message coordination earns        %.3f   (coordination closes +%.3f, collisions -> %.3f)"
          % (cc, cc - sf, res["comm_cf"]["coll"][0]))
    print("  remaining gap to ceiling (estimation)    %.3f" % (1.0 - cc))

    import json, time
    out_dir = os.path.join(ROOT, "results", "pilots")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "partial_comms_%s.json" % time.strftime("%Y%m%d_%H%M%S"))
    json.dump({"meta": {"experiment": "partial-communication decomposition of the ceiling gap",
                        "algorithms": ALGOS, "seeds": SEEDS, "offer_size": 0, "capacity_one": True,
                        "rho": 0.5, "metric": "earned (frac of capacity-1 Hungarian ceiling) + collision rate"},
               "results": res, "raw": raw}, open(out_path, "w"), indent=1)
    print("\nsaved ->", out_path)


if __name__ == "__main__":
    main()
