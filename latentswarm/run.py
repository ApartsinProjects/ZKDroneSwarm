"""Config-driven LatentSwarm runner. Builds the scenario, env, policies, and metrics from a
RunConfig, runs one T-round mission per seed per policy, and reports earned (anytime) skill and
unseen-pair skill with bootstrap 95% CIs.

Usage:
    python -m latentswarm.run                # default config
    python -m latentswarm.run --out results/pilots/latentswarm_main.json
"""
import argparse
import json
import os

import numpy as np

from .config import RunConfig
from .registry import ALGORITHMS, get
from .scenarios import build_scenario
from .env import ZKMRTAEnv
from .metrics import hungarian_oracle_per_step, UnseenPairSkill, EarnedSkill, bootstrap_ci


def run_mission(env: ZKMRTAEnv, policy, T: int):
    obs = env.reset()
    per_round = []
    for _ in range(T):
        actions = policy.act(obs)
        obs, rewards, _ = env.step(actions)
        policy.observe(obs)
        per_round.append(float(np.mean(rewards)))
    return per_round, env.engaged


def run(cfg: RunConfig):
    algos = list(dict.fromkeys(["random"] + list(cfg.algorithms)))   # always include random
    earned = {a: [] for a in algos + ["oracle"]}
    unseen = {a: [] for a in algos + ["oracle"]}
    earned_metric, unseen_metric = EarnedSkill(), UnseenPairSkill()
    d_guesses = []
    for s in cfg.seeds:
        world_rng = np.random.RandomState(s)
        P, U = build_scenario(cfg, world_rng).generate()
        d_guess = cfg.rank_for_run(world_rng)
        d_guesses.append(d_guess)
        oracle_step = hungarian_oracle_per_step(P, U)

        runs = {}
        for a in algos:
            env = ZKMRTAEnv(cfg, P, U, d_guess, seed=s)
            pol = get(ALGORITHMS, a)(cfg, cfg.m, cfg.n, d_guess, seed=1000 + s)
            per_round, engaged = run_mission(env, pol, cfg.T)
            runs[a] = (float(np.mean(per_round)), pol.predict_rows(), engaged)

        rnd_mean = runs["random"][0]
        useed = np.random.RandomState(10000 + s)
        for a in algos:
            mean_r, pred, engaged = runs[a]
            earned[a].append(earned_metric.compute(mean_reward=mean_r, random_mean=rnd_mean, oracle_mean=oracle_step))
            unseen[a].append(unseen_metric.compute(P=P, U=U, pred_rows=pred, engaged=engaged, rng=useed))
        earned["oracle"].append(1.0)
        unseen["oracle"].append(1.0)
        print("seed %2d (d_hat=%d): " % (s, d_guess)
              + "  ".join("%s e=%.3f u=%s" % (a, earned[a][-1],
                          ("%.3f" % unseen[a][-1]) if unseen[a][-1] is not None else "na") for a in algos))

    print("\n%-12s | earned skill         | unseen-pair skill" % "policy")
    print("-" * 60)
    summary = {}
    for a in algos + ["oracle"]:
        em, elo, ehi = bootstrap_ci(earned[a])
        um, ulo, uhi = bootstrap_ci(unseen[a])
        summary[a] = {"earned": (em, elo, ehi), "unseen": (um, ulo, uhi)}
        us = "%.3f [%.3f, %.3f]" % (um, ulo, uhi) if um is not None else "n/a"
        print("%-12s | %.3f [%.3f, %.3f] | %s" % (a, em, elo, ehi, us))
    print("-" * 60)
    return {"meta": {"d_guesses": d_guesses, "config": cfg.to_dict()},
            "earned": earned, "unseen": unseen, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pilots/latentswarm_main.json")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = RunConfig.load(args.config) if args.config else RunConfig(
        scenario="uniform_cosine",
        algorithms=["random", "ucb_indep", "mf_sgd", "club", "bias_model", "swarm_cf"])
    out = run(cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=0)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
