"""LatentSwarm contention study: the cost of NO de-confliction under capacity-1 contention.

Under the all-tasks menu (offer_size=0) every robot is offered the same full set, so uncoordinated
greedy policies converge on the same high-value tasks and COLLIDE (only the first robot to pick a
task each round succeeds). We report, per policy: earned (anytime) skill AND the collision rate
(fraction of robots whose pick collided), and how restricting the menu (offer_size=20) changes both.
No de-confliction mechanism is used here (a communication-free de-confliction scheme is deferred to
the follow-up). Reuses the LatentSwarm package verbatim; seeds/scenario/d_hat match latentswarm.run.

Saves results/pilots/latentswarm_contention_<stamp>.json. Run from anywhere.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)                                   # for the latentswarm package
sys.path.insert(0, os.path.join(ROOT, "experiments"))     # for _results_io
sys.stdout.reconfigure(encoding="utf-8")

from latentswarm.config import RunConfig
from latentswarm.registry import ALGORITHMS, get
from latentswarm.scenarios import build_scenario
from latentswarm.env import ZKMRTAEnv
from latentswarm.metrics import hungarian_oracle_per_step, EarnedSkill, bootstrap_ci
from _results_io import save_results

ALGOS = ["random", "ucb_indep", "mf_sgd", "swarm_cf"]
OFFERS = [0, 20]                       # 0 = all tasks (default); 20 = size-c menu
SEEDS = list(range(16))


def run_one(cfg, P, U, d_guess, seed, algo):
    """One mission; returns (mean per-round reward, mean collision rate over the mission)."""
    env = ZKMRTAEnv(cfg, P, U, d_guess, seed=seed)
    pol = get(ALGORITHMS, algo)(cfg, cfg.m, cfg.n, d_guess, seed=1000 + seed)
    obs = env.reset()
    rew, coll = [], []
    for _ in range(cfg.T):
        actions = pol.act(obs)
        obs, rewards, info = env.step(actions)
        pol.observe(obs)
        rew.append(float(np.mean(rewards)))
        coll.append(info["collisions"] / cfg.m)            # fraction of robots that collided
    return float(np.mean(rew)), float(np.mean(coll))


def main():
    earned_metric = EarnedSkill()
    raw = {str(c): {a: {"earned": [], "coll": []} for a in ALGOS} for c in OFFERS}
    for c in OFFERS:
        for s in SEEDS:
            wr = np.random.RandomState(s)                  # SAME world construction as latentswarm.run
            cfg = RunConfig(offer_size=c, seeds=[s])
            P, U = build_scenario(cfg, wr).generate()
            d_guess = cfg.rank_for_run(wr)
            oracle = hungarian_oracle_per_step(P, U)
            rnd_mean, _ = run_one(cfg, P, U, d_guess, s, "random")
            for a in ALGOS:
                er, cr = run_one(cfg, P, U, d_guess, s, a)
                sk = earned_metric.compute(mean_reward=er, random_mean=rnd_mean, oracle_mean=oracle)
                raw[str(c)][a]["earned"].append(sk)
                raw[str(c)][a]["coll"].append(cr)

    print("=" * 78)
    print("LatentSwarm contention: earned skill + collision rate (capacity-1, persistent, "
          "rho=0.5, %d seeds)" % len(SEEDS))
    print("=" * 78)
    for c in OFFERS:
        menu = "ALL tasks (c=n)" if c == 0 else ("size-c menu (c=%d)" % c)
        print("\n-- %s --" % menu)
        print("%-12s | %-22s | %-22s" % ("policy", "earned skill", "collision rate"))
        for a in ALGOS:
            em, elo, ehi = bootstrap_ci(raw[str(c)][a]["earned"])
            cm, clo, chi = bootstrap_ci(raw[str(c)][a]["coll"])
            print("%-12s | %6.3f [%6.3f,%6.3f] | %6.3f [%6.3f,%6.3f]"
                  % (a, em, elo, ehi, cm, clo, chi))
    print("\nREAD: under the all-tasks menu, structure-free Independent-UCB collides most and earns")
    print("worst (often <0); restricting the menu cuts collisions. SwarmCF earns most at both menus.")
    print("No de-confliction is used (deferred to the follow-up).")

    path = save_results("latentswarm_contention", {
        "meta": {"experiment": "LatentSwarm contention: earned + collision rate vs offer size",
                 "algorithms": ALGOS, "offers": OFFERS, "seeds": SEEDS,
                 "capacity_one": True, "mask_mode": "persistent", "rho": 0.5,
                 "metric": "earned (anytime) skill + collision rate (frac robots collided), "
                           "no de-confliction (follow-up)"},
        "raw": raw}, results_dir=os.path.join(ROOT, "results", "pilots"))
    print("complete data saved ->", path)


if __name__ == "__main__":
    main()
