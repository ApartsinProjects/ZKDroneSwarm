"""
Principled measurement of effective concentration s_eff for Theorem 11'.

Theorem 11' states that the expected pairwise collision count per step is
    E[C(t)] = m(m-d) / (2 * d * s),
where s is the effective concentration (number of targets each drone
distributes mass over). Inverting this:
    s_eff = m(m-d) / (2 * d * E[C(t)])

This script re-runs MF-CF and PTF at the n=27, d=3 baseline with a wrapper
that logs per-step actions per drone, then computes the empirical action
distribution per drone and derives s_eff in two ways:
  1. Inverse participation ratio (IPR): s_eff^IPR = 1 / sum_j p_ij^2
  2. Collision-rate inversion (T11'): s_eff^T11 = m(m-d) / (2 d * collisions/step)
"""

import json, os, sys, statistics, glob
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tabula_drone.config import load_config
from tabula_drone.utils.metrics_manager import EpisodeMetrics, MetricsManager
from tabula_drone.policies.base import bind_diagnostics_provider
from experiments.env_setup import build_env_and_configs
from experiments.run_experiment_sweep import build_policy, apply_overrides


def run_with_action_log(policy_name: str, seed: int, episodes: int = 35,
                        n: int = 27, d: int = 3):
    """Run one full sweep and accumulate per-drone action counts per episode."""
    config = load_config("config/scenario.json")
    import dataclasses
    config = dataclasses.replace(config, seed=seed)
    overrides = {"num_targets": n}
    if d != 3:
        overrides["env_latent_dim"] = d
    extra = {}
    config, extra = apply_overrides(config, overrides, extra)

    env, drones_config, targets_config, builder = build_env_and_configs(config, seed=seed)
    num_agents = env.num_drones
    num_targets = env.num_targets
    policy = build_policy(policy_name, config, num_agents, num_targets, extra)

    # Per-episode per-drone action count matrix (n_agents x n_targets)
    ep_counts = []

    for ep in range(1, episodes + 1):
        ep_seed = seed + ep - 1
        obs, infos = env.reset(seed=ep_seed)
        bind_diagnostics_provider(policy, lambda: env.diagnostics.to_dict() if env.diagnostics else {})
        action_counts = np.zeros((num_agents, num_targets), dtype=np.int64)

        done = False
        while not done:
            try:
                actions = policy.select_actions(obs, infos, env=env)
            except TypeError:
                actions = policy.select_actions(obs, infos)
            # Record actions (1-indexed; 0 = noop)
            for i, drone_id in enumerate(sorted(actions.keys())):
                a = int(actions[drone_id])
                if a > 0 and a - 1 < num_targets:
                    action_counts[i, a - 1] += 1
            obs, rewards, terminations, truncations, infos = env.step(actions)
            policy.update(obs)
            ref = env.agents[0]
            done = terminations[ref] or truncations[ref]

        ep_counts.append(action_counts)
        policy.soft_reset()

    return ep_counts


def s_eff_from_counts(counts: np.ndarray) -> float:
    """Inverse participation ratio averaged over drones."""
    p = counts.astype(np.float64)
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)
    p = p / row_sums  # normalise per-drone
    ipr = 1.0 / np.maximum(np.sum(p ** 2, axis=1), 1e-12)  # per drone
    return float(np.mean(ipr))


def collisions_per_step(counts: np.ndarray, ep_steps: int) -> float:
    """
    Estimate expected collisions per step from action counts.

    For each target j, the contribution to expected collisions is
    C(j) = sum over distinct drone pairs (i,i') of (count_ij * count_i'j) / ep_steps^2
    Total per step = sum_j C(j) * ep_steps
    """
    m, n = counts.shape
    p = counts.astype(np.float64) / max(ep_steps, 1)  # P[a=j per step] approx
    # E[C(t)] per step = sum_{i<i'} sum_j p_ij p_i'j
    total = 0.0
    for j in range(n):
        col = p[:, j]
        sm = float(np.sum(col))
        sm2 = float(np.sum(col ** 2))
        total += 0.5 * (sm * sm - sm2)
    return total


if __name__ == "__main__":
    seeds = [42, 123, 456, 789, 1337]
    print("s_eff measurement for MF-CF and PTF K=5 at n=27, d=3, 5 seeds")
    print("=" * 75)
    print(f"{'Policy':12s}  {'EpRange':>9s}  {'avg colls/step':>14s}  {'s_eff^IPR':>10s}  {'s_eff^T11':>10s}")
    print("-" * 75)

    m, d, n = 9, 3, 27
    factor = m * (m - d) / (2 * d)  # = 9*6/6 = 9

    for policy in ["mf_cf", "ptf_k5"]:
        # Aggregate across seeds; report by phase (early, mid, late)
        # Phase 1 (eps 1-9), Phase 2 (eps 9-21), Phase 3 (eps 22-35)
        agg = {"ep1_9": [], "ep10_21": [], "ep22_35": [], "all": []}
        coll_agg = {k: [] for k in agg}

        for seed in seeds:
            ep_counts = run_with_action_log(policy, seed, episodes=35, n=n, d=d)
            for ep_idx, cnt in enumerate(ep_counts):
                ep_steps = int(cnt.sum() / m) if cnt.sum() > 0 else 1
                seff = s_eff_from_counts(cnt)
                coll = collisions_per_step(cnt, ep_steps)
                ep_num = ep_idx + 1
                agg["all"].append(seff)
                coll_agg["all"].append(coll)
                if ep_num <= 9:
                    agg["ep1_9"].append(seff)
                    coll_agg["ep1_9"].append(coll)
                elif ep_num <= 21:
                    agg["ep10_21"].append(seff)
                    coll_agg["ep10_21"].append(coll)
                else:
                    agg["ep22_35"].append(seff)
                    coll_agg["ep22_35"].append(coll)

        for phase in ["ep1_9", "ep10_21", "ep22_35", "all"]:
            if agg[phase] and coll_agg[phase]:
                seff_mean = statistics.mean(agg[phase])
                coll_mean = statistics.mean(coll_agg[phase])
                s_t11 = factor / max(coll_mean, 1e-6)
                label = {"ep1_9": "eps 1-9", "ep10_21": "eps 10-21",
                         "ep22_35": "eps 22-35", "all": "all 35"}[phase]
                print(f"{policy:12s}  {label:>9s}  {coll_mean:>14.4f}  "
                      f"{seff_mean:>10.3f}  {s_t11:>10.3f}")

    print()
    print(f"Reference: factor m(m-d)/(2d) = {factor:.1f}")
    print(f"  s = 1 => collisions = {factor:.1f}/step  (peak crowding, Theorem 11')")
    print(f"  s = n/d = {n/d:.0f} => collisions = {factor/(n/d):.2f}/step (perfect matched)")
