"""
Wave 3b-2: validate our method in the REAL tabula_drone simulator (spatial targets,
HP depletion, episodic). Build the env per outer seed, run each policy through a
sequence of learning episodes, and score by converged episode reward, normalized to
skill = (policy - random) / (oracle - random). Compares our WeightedALS (= RewardCF
in the env interface) vs the env's SGD matrix factorization, UCBIndep, random, and
the centralized oracle.

Run from the repo root:  python experiments/tabula_bench.py
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from tabula_drone.config import load_config
from tabula_drone.envs.drone_engage_latent_mrta import DroneEngageLatentMRTA
from tabula_drone.scenarios.latent_scenario_builder import LatentScenarioBuilder
from tabula_drone.policies.random_policy import RandomPolicy
from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy
from tabula_drone.policies.ucb_indep_policy import UCBIndepPolicy
from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
from tabula_drone.policies.weighted_als_policy import WeightedALSPolicy

CONFIG = "config/scenario.json"
SEEDS = [0, 1, 2]
EPISODES = 16


def build_env(config, seed):
    b = LatentScenarioBuilder(world_size=config.world.size, config=config.latent_world,
                              target_hp=config.targets.target_hp, seed=seed)
    b.with_drones(count=config.drones.count, region=config.drones.region,
                  min_distance_between_drones=config.drones.min_distance_between_drones)
    b.with_targets(count=config.targets.count, region=config.targets.region,
                   min_distance_from_drones=config.targets.min_distance_from_drones,
                   min_distance_between_targets=config.targets.min_distance_between_targets)
    dcfg, tcfg = b.build()
    cf = config.collaborative_filtering
    rn = cf.matrix_factorization_cf.reward_noise if (cf and cf.matrix_factorization_cf and cf.matrix_factorization_cf.reward_noise is not None) \
        else (cf.reward_noise if cf and cf.reward_noise is not None else 0.0)
    on = cf.observation_noise if cf and cf.observation_noise is not None else 0.0
    env = DroneEngageLatentMRTA(world_size=config.world.size, max_steps=config.environment.max_steps,
                                drones_config=dcfg, targets_config=tcfg,
                                scenario_id=config.environment.scenario_id, reward_noise=rn,
                                observation_noise=on, builder=b,
                                latent_world={"mode": b.mode, "latent_dim": b.latent_dim}, target_hp=config.targets.target_hp)
    return env, len(dcfg), len(tcfg)


def make_policy(ptype, config, m, n, seed):
    if ptype == "random":
        return RandomPolicy(seed=seed, allow_noop=config.policy.allow_noop)
    if ptype == "oracle":
        return OptimalAssignmentOracle(seed=seed, allow_noop=config.policy.allow_noop)
    if ptype == "ucb_indep":
        return UCBIndepPolicy(num_agents=m, num_targets=n, c=2.0, seed=seed, allow_noop=config.policy.allow_noop)
    if ptype == "mf":
        mf = config.collaborative_filtering.matrix_factorization_cf
        pol = {f"drone_{i}": MatrixFactorizationPolicy(num_targets=n, agent_idx=i, num_agents=m,
               latent_dim=mf.latent_dim or 8, learning_rate=mf.learning_rate or 0.01,
               lambda_reg=mf.lambda_reg if mf.lambda_reg is not None else 0.02,
               epsilon=0.3, epsilon_decay=0.9995, epsilon_min=0.02,
               anti_signal_weight=mf.anti_signal_weight if mf.anti_signal_weight is not None else 1.0,
               use_integration_matrix=bool(mf.use_integration_matrix), seed=seed + i) for i in range(m)}
        return MultiAgentPolicy(pol)
    if ptype == "weighted_als":
        pol = {f"drone_{i}": WeightedALSPolicy(num_targets=n, agent_idx=i, num_agents=m,
               latent_dim=8, lam=1.0, als_sweeps=8, refit_every=8,
               epsilon=0.3, epsilon_decay=0.9995, epsilon_min=0.02, seed=seed + i) for i in range(m)}
        return MultiAgentPolicy(pol)
    raise ValueError(ptype)


def run_episode(env, policy, seed):
    obs, infos = env.reset(seed=seed)
    ref = env.agents[0]; tot = 0.0; steps = 0; done = False
    while not done:
        try:
            actions = policy.select_actions(obs, infos, env=env)
        except TypeError:
            actions = policy.select_actions(obs, infos)
        obs, rewards, term, trunc, infos = env.step(actions)
        try:
            policy.update(obs)
        except Exception:
            pass
        tot += float(sum(rewards.values())); steps += 1
        done = bool(term[ref]) or bool(trunc[ref])
    if hasattr(policy, "soft_reset"):
        policy.soft_reset()
    # EFFICIENCY metric (mean reward per step): rewards good matches / few wasted
    # shots; the oracle (finishes fast, high-quality engagements) is the ceiling.
    return tot / max(steps, 1)


def main():
    config = load_config(CONFIG)
    ptypes = ["random", "oracle", "ucb_indep", "mf", "weighted_als"]
    raw = {p: [] for p in ptypes}            # per-seed converged mean reward
    traj = {p: [] for p in ptypes}           # per-seed per-episode reward (for curves)
    for s in SEEDS:
        env, m, n = build_env(config, s)
        print("seed %d: m=%d n=%d" % (s, m, n))
        for p in ptypes:
            pol = make_policy(p, config, m, n, seed=1000 + s)
            ep = [run_episode(env, pol, seed=s * 100 + e) for e in range(EPISODES)]
            traj[p].append(ep)
            raw[p].append(float(np.mean(ep[EPISODES // 2:])))   # converged (2nd half)
            print("  %-13s converged reward = %8.2f" % (p, raw[p][-1]))
    rnd = np.array(raw["random"]); orc = np.array(raw["oracle"])
    print("\n%-13s | conv reward (mean) | skill = (p-rand)/(orac-rand)" % "policy")
    print("-" * 64)
    skills = {}
    for p in ptypes:
        v = np.array(raw[p]); sk = (v - rnd) / np.maximum(orc - rnd, 1e-9)
        skills[p] = sk.tolist()
        print("%-13s | %18.2f | %.3f +- %.3f" % (p, v.mean(), sk.mean(), sk.std()))
    print("-" * 64)
    print("Validation: weighted_als (ours) should beat mf / ucb_indep and approach oracle (skill->1).")
    os.makedirs("results/pilots", exist_ok=True)
    out = "results/pilots/tabula_bench_real.json"
    json.dump({"meta": {"experiment": "real tabula_drone env validation", "seeds": SEEDS,
                        "episodes": EPISODES, "policies": ptypes,
                        "metric": "converged episode reward + skill vs random/oracle"},
               "raw_reward": raw, "skill": skills, "traj": traj}, open(out, "w"), indent=0)
    print("saved ->", out)


if __name__ == "__main__":
    main()
