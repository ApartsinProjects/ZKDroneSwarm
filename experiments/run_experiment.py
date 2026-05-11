"""
experiments/run_experiment.py

Run a single policy for one seed and write results to a JSON file.

Usage:
    python experiments/run_experiment.py --policy random --seed 42 --out results/random_s42.json
    python experiments/run_experiment.py --policy oracle_l --seed 42 --out results/oracle_l_s42.json
    python experiments/run_experiment.py --policy ucb_indep --seed 42 --episodes 35 --out results/ucb_s42.json

Supported policy names: random, oracle_hp, oracle_l, ucb_indep, mf_cf
"""

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tabula_drone.config import load_config
from tabula_drone.utils.metrics_manager import EpisodeMetrics, MetricsManager
from tabula_drone.policies.base import bind_diagnostics_provider
from experiments.env_setup import build_env_and_configs

CONFIG_PATH = "config/scenario.json"


def get_env_diagnostics(env):
    if env.diagnostics is None:
        return {}
    return env.diagnostics.to_dict()


def build_policy(policy_name, config, num_agents, num_targets):
    from tabula_drone.policies.random_policy import RandomPolicy
    from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
    from tabula_drone.policies.ucb_indep_policy import UCBIndepPolicy
    from tabula_drone.policies.oracle_l_policy import OracleLPolicy
    from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy
    from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy

    allow_noop = config.policy.allow_noop
    seed = config.seed

    if policy_name == "random":
        return RandomPolicy(seed=seed, allow_noop=allow_noop)
    elif policy_name == "oracle_hp":
        return OptimalAssignmentOracle(seed=seed, allow_noop=allow_noop)
    elif policy_name == "oracle_l":
        return OracleLPolicy(allow_noop=allow_noop)
    elif policy_name == "ucb_indep":
        return UCBIndepPolicy(
            num_agents=num_agents, num_targets=num_targets,
            c=2.0, seed=seed, allow_noop=allow_noop,
        )
    elif policy_name == "mf_cf":
        mf_cfg = config.collaborative_filtering.matrix_factorization_cf
        policies = {}
        for i in range(num_agents):
            policies[f"drone_{i}"] = MatrixFactorizationPolicy(
                num_targets=num_targets, agent_idx=i, num_agents=num_agents,
                latent_dim=mf_cfg.latent_dim, learning_rate=mf_cfg.learning_rate,
                lambda_reg=mf_cfg.lambda_reg, epsilon=mf_cfg.epsilon,
                epsilon_decay=mf_cfg.epsilon_decay, epsilon_min=mf_cfg.epsilon_min,
                anti_signal_weight=mf_cfg.anti_signal_weight,
                use_integration_matrix=bool(mf_cfg.use_integration_matrix),
                seed=seed + i if seed is not None else None,
            )
        return MultiAgentPolicy(policies)
    else:
        raise ValueError(f"Unknown policy: {policy_name!r}")


def run_one_episode(env, policy, episode_num, seed):
    obs, infos = env.reset(seed=seed)
    bind_diagnostics_provider(policy, lambda: get_env_diagnostics(env))

    total_rewards = {a: 0.0 for a in env.agents}
    step_count = 0
    done = False
    total_net_damage = 0.0
    total_gross_damage = 0.0
    total_collisions = 0
    total_latent_mismatch = 0.0
    total_optimal_potential = 0.0
    overkill_events = []
    shared_info = {}

    while not done:
        step_count += 1
        try:
            actions = policy.select_actions(obs, infos, env=env)
        except TypeError:
            actions = policy.select_actions(obs, infos)

        obs, rewards, terminations, truncations, infos = env.step(actions)
        shared_info = get_env_diagnostics(env)
        policy.update(obs)

        ref = env.agents[0]
        done = terminations[ref] or truncations[ref]
        for a in env.agents:
            total_rewards[a] += rewards[a]
        total_net_damage += shared_info.get("net_damage", 0.0)
        total_gross_damage += shared_info.get("total_gross_damage", 0.0)
        total_latent_mismatch += shared_info.get("latent_mismatch", 0.0)
        total_optimal_potential += shared_info.get("optimal_potential", 0.0)
        total_collisions += shared_info.get("collisions", 0)
        if "overkill" in shared_info:
            overkill_events.append(shared_info["overkill"])

    ammo_used_dict = shared_info.get("ammo_used", {})
    total_ammo_used = sum(ammo_used_dict.values())
    total_overkill = sum(sum(ev.values()) for ev in overkill_events)
    targets_neutralized = shared_info.get("cumulative_neutralizations", 0)
    done_reason = shared_info.get("done_reason")

    return EpisodeMetrics(
        episode=episode_num,
        steps=step_count,
        done_reason=done_reason,
        targets_neutralized=targets_neutralized,
        total_ammo_used=total_ammo_used,
        total_overkill=total_overkill,
        total_net_damage=total_net_damage,
        total_gross_damage=total_gross_damage,
        total_collisions=total_collisions,
        agent_rewards=total_rewards,
        weapon_damage_profile_mapping=env.weapon_damage_profile_mapping,
        total_latent_mismatch=total_latent_mismatch,
        total_optimal_potential=total_optimal_potential,
    )


def metrics_to_dict(m):
    return {
        "episode": m.episode,
        "steps": m.steps,
        "done_reason": m.done_reason,
        "targets_neutralized": m.targets_neutralized,
        "total_ammo_used": m.total_ammo_used,
        "total_overkill": round(m.total_overkill, 4),
        "total_net_damage": round(m.total_net_damage, 4),
        "total_gross_damage": round(m.total_gross_damage, 4),
        "total_collisions": m.total_collisions,
        "total_reward": round(m.total_reward, 6),
        "avg_latent_match_quality": round(m.avg_latent_match_quality, 6),
        "shots_per_target": round(m.shots_per_target or 0.0, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True,
                        choices=["random", "oracle_hp", "oracle_l", "ucb_indep", "mf_cf"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--out", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    import dataclasses
    config = dataclasses.replace(config, seed=args.seed)

    env, drones_config, targets_config, builder = build_env_and_configs(config, seed=args.seed)
    num_agents = env.num_drones
    num_targets = env.num_targets

    policy = build_policy(args.policy, config, num_agents, num_targets)

    is_det = getattr(policy, "is_deterministic", False)
    if args.episodes is not None:
        num_episodes = args.episodes
    elif is_det:
        num_episodes = 1
    else:
        num_episodes = config.environment.episodic.num_episodes

    print(f"Running {args.policy} seed={args.seed} for {num_episodes} episode(s)...")

    episode_metrics = []
    for ep in range(1, num_episodes + 1):
        ep_seed = args.seed + ep - 1
        m = run_one_episode(env, policy, ep, ep_seed)
        episode_metrics.append(m)
        policy.soft_reset()
        if args.verbose or ep % 5 == 0:
            print(f"  ep={ep:3d}  steps={m.steps:4d}  neutralized={m.targets_neutralized:2d}"
                  f"  lmq={m.avg_latent_match_quality:.4f}  reason={m.done_reason}")

    mgr = MetricsManager()
    summary = mgr.calc_total_episodes_metrics(episode_metrics)

    output = {
        "policy": args.policy,
        "seed": args.seed,
        "num_episodes": num_episodes,
        "episodes": [metrics_to_dict(m) for m in episode_metrics],
        "summary": {
            "avg_steps": round(summary.avg_steps, 2),
            "avg_targets": round(summary.avg_targets, 4),
            "avg_ammo": round(summary.avg_ammo, 4),
            "avg_overkill": round(summary.avg_overkill, 4),
            "avg_reward": round(summary.avg_reward, 6),
            "avg_net_damage": round(summary.avg_net_damage, 4),
            "avg_gross_damage": round(summary.avg_gross_damage, 4),
            "success_count": summary.success_count,
            "success_rate": round(summary.success_rate, 2),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved to {args.out}")
    print(f"Summary: avg_steps={summary.avg_steps:.1f}  avg_targets={summary.avg_targets:.2f}"
          f"  success_rate={summary.success_rate:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
