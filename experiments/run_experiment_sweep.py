"""
experiments/run_experiment_sweep.py

Like run_experiment.py but accepts --override-json to patch config values.
Used by run_sweep.py for ablation conditions.

Override keys supported:
  latent_dim           -> collaborative_filtering.matrix_factorization_cf.latent_dim
  reward_noise         -> collaborative_filtering.reward_noise
  observation_noise    -> collaborative_filtering.observation_noise
  use_integration_matrix -> collaborative_filtering.matrix_factorization_cf.use_integration_matrix
  ucb_c                -> UCBIndepPolicy c parameter (passed directly)

Usage:
    python experiments/run_experiment_sweep.py \
        --policy mf_cf --seed 42 --out results/sweep/mf_latent5_s42.json \
        --episodes 35 --override-json '{"latent_dim": 5}'
"""

import argparse
import copy
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


def apply_overrides(config, overrides, extra):
    """Apply override dict to config. Returns (config, extra) where extra passes
    non-config overrides (like ucb_c) to the policy builder.

    Supported override keys:
      latent_dim           -> MF-CF factorization dimension
      use_integration_matrix -> MF-CF supervision mode
      reward_noise         -> reward noise level
      observation_noise    -> observation noise level
      ucb_c                -> UCB exploration constant
      num_targets          -> number of targets in the scenario
      num_drones           -> number of drones in the scenario
      env_latent_dim       -> environment latent dimension AND num_modes
                             (sets both latent_world.latent_dim and all
                              num_modes fields so they stay consistent)
    """
    import dataclasses

    cf = config.collaborative_filtering
    mf = cf.matrix_factorization_cf if cf else None
    ucb_c = extra.get("ucb_c", 2.0)

    # --- MF-CF config overrides ---
    if "latent_dim" in overrides and mf is not None:
        mf = dataclasses.replace(mf, latent_dim=overrides["latent_dim"])
    if "use_integration_matrix" in overrides and mf is not None:
        mf = dataclasses.replace(mf, use_integration_matrix=bool(overrides["use_integration_matrix"]))
    if "reward_noise" in overrides and cf is not None:
        cf = dataclasses.replace(cf, reward_noise=overrides["reward_noise"])
    if "observation_noise" in overrides and cf is not None:
        cf = dataclasses.replace(cf, observation_noise=overrides["observation_noise"])
    if "effect_noise" in overrides and cf is not None:
        cf = dataclasses.replace(cf, effect_noise=overrides["effect_noise"])
    if "ucb_c" in overrides:
        ucb_c = float(overrides["ucb_c"])

    # --- IQL-ZK overrides (revision item 3) ---
    if "iql_alpha" in overrides:
        extra["iql_alpha"] = float(overrides["iql_alpha"])
    if "iql_epsilon_decay" in overrides:
        extra["iql_epsilon_decay"] = float(overrides["iql_epsilon_decay"])

    # --- Latent-world center_mode overrides (revision item 6) ---
    # The center_mode field lives on latent_world directly (not on drones/targets
    # sub-configs). Valid values: "random", "orthogonal", "one_hot".
    if "center_mode" in overrides:
        lw = config.latent_world
        new_lw = dataclasses.replace(lw, center_mode=str(overrides["center_mode"]))
        config = dataclasses.replace(config, latent_world=new_lw)

    if mf is not None and cf is not None:
        cf = dataclasses.replace(cf, matrix_factorization_cf=mf)
    if cf is not None:
        config = dataclasses.replace(config, collaborative_filtering=cf)

    # --- Scenario structure overrides ---
    if "num_targets" in overrides:
        new_targets = dataclasses.replace(config.targets, count=int(overrides["num_targets"]))
        config = dataclasses.replace(config, targets=new_targets)

    if "num_drones" in overrides:
        new_drones = dataclasses.replace(config.drones, count=int(overrides["num_drones"]))
        config = dataclasses.replace(config, drones=new_drones)

    if "max_steps" in overrides:
        new_env = dataclasses.replace(config.environment, max_steps=int(overrides["max_steps"]))
        config = dataclasses.replace(config, environment=new_env)

    if "env_latent_dim" in overrides:
        # Change the environment's latent dimension.  This must update:
        #   latent_world.latent_dim
        #   latent_world.drones.num_modes
        #   latent_world.targets.num_modes
        # so the builder generates latent vectors in the right space.
        d = int(overrides["env_latent_dim"])
        lw = config.latent_world
        new_drone_lw = dataclasses.replace(lw.drones, num_modes=d)
        new_target_lw = dataclasses.replace(lw.targets, num_modes=d)
        new_lw = dataclasses.replace(lw, latent_dim=d,
                                     drones=new_drone_lw, targets=new_target_lw)
        config = dataclasses.replace(config, latent_world=new_lw)
        # Also update MF-CF's latent_dim to match unless explicitly set
        if "latent_dim" not in overrides:
            cf = config.collaborative_filtering
            mf = cf.matrix_factorization_cf if cf else None
            if mf is not None:
                mf = dataclasses.replace(mf, latent_dim=d)
                cf = dataclasses.replace(cf, matrix_factorization_cf=mf)
                config = dataclasses.replace(config, collaborative_filtering=cf)

    extra["ucb_c"] = ucb_c
    return config, extra


def build_policy(policy_name, config, num_agents, num_targets, extra):
    from tabula_drone.policies.random_policy import RandomPolicy
    from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
    from tabula_drone.policies.ucb_indep_policy import UCBIndepPolicy
    from tabula_drone.policies.ucb_homogeneous_policy import UCBHomogeneousPolicy
    from tabula_drone.policies.oracle_l_policy import OracleLPolicy
    from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy
    from tabula_drone.policies.no_broadcast_matrix_factorization_policy import NoBroadcastMatrixFactorizationPolicy
    from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
    from tabula_drone.policies.probe_then_fit_policy import ProbeThenFitPolicy
    from tabula_drone.policies.iql_zk_policy import IQLZKPolicy
    from tabula_drone.policies.estr_policy import ESTRPolicy
    from tabula_drone.policies.thompson_mf_policy import ThompsonMFPolicy

    allow_noop = config.policy.allow_noop
    seed = config.seed
    ucb_c = extra.get("ucb_c", 2.0)

    if policy_name == "random":
        return RandomPolicy(seed=seed, allow_noop=allow_noop)
    elif policy_name == "oracle_hp":
        return OptimalAssignmentOracle(seed=seed, allow_noop=allow_noop)
    elif policy_name == "oracle_l":
        return OracleLPolicy(allow_noop=allow_noop)
    elif policy_name == "ucb_indep":
        return UCBIndepPolicy(
            num_agents=num_agents, num_targets=num_targets,
            c=ucb_c, seed=seed, allow_noop=allow_noop,
        )
    elif policy_name == "ucb_homo":
        return UCBHomogeneousPolicy(
            num_agents=num_agents, num_targets=num_targets,
            c=ucb_c, seed=seed, allow_noop=allow_noop,
        )
    elif policy_name == "iql_zk":
        return IQLZKPolicy(
            num_agents=num_agents, num_targets=num_targets,
            alpha=extra.get("iql_alpha", 0.1),
            gamma=0.0,
            epsilon=0.30,
            epsilon_decay=extra.get("iql_epsilon_decay", 0.9995),
            epsilon_min=0.05,
            seed=seed, allow_noop=allow_noop,
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
    elif policy_name == "mf_cf_nb":
        # No-broadcast MF-CF: each drone learns ONLY from its own observation.
        # Implements the empirical verification of Theorem 6 (broadcast multiplier).
        mf_cfg = config.collaborative_filtering.matrix_factorization_cf
        policies = {}
        for i in range(num_agents):
            policies[f"drone_{i}"] = NoBroadcastMatrixFactorizationPolicy(
                num_targets=num_targets, agent_idx=i, num_agents=num_agents,
                latent_dim=mf_cfg.latent_dim, learning_rate=mf_cfg.learning_rate,
                lambda_reg=mf_cfg.lambda_reg, epsilon=mf_cfg.epsilon,
                epsilon_decay=mf_cfg.epsilon_decay, epsilon_min=mf_cfg.epsilon_min,
                anti_signal_weight=mf_cfg.anti_signal_weight,
                use_integration_matrix=bool(mf_cfg.use_integration_matrix),
                seed=seed + i if seed is not None else None,
            )
        return MultiAgentPolicy(policies)
    elif policy_name == "estr":
        # Centralised low-rank bandit baseline (Kang-Hsieh-Lee 2022 style,
        # ablated to public broadcast). Explore for ~5 episodes then commit.
        mf_cfg = config.collaborative_filtering.matrix_factorization_cf
        explore_steps = int(extra.get("estr_explore_steps", 350))
        return ESTRPolicy(
            num_agents=num_agents, num_targets=num_targets,
            latent_dim=mf_cfg.latent_dim,
            explore_steps=explore_steps,
            seed=seed, allow_noop=allow_noop,
        )
    elif policy_name == "ts_mf":
        # Thompson-sampling variant of MF-CF.
        mf_cfg = config.collaborative_filtering.matrix_factorization_cf
        policies = {}
        for i in range(num_agents):
            policies[f"drone_{i}"] = ThompsonMFPolicy(
                num_targets=num_targets, agent_idx=i, num_agents=num_agents,
                latent_dim=mf_cfg.latent_dim, learning_rate=mf_cfg.learning_rate,
                lambda_reg=mf_cfg.lambda_reg, epsilon=mf_cfg.epsilon,
                epsilon_decay=mf_cfg.epsilon_decay, epsilon_min=mf_cfg.epsilon_min,
                anti_signal_weight=mf_cfg.anti_signal_weight,
                use_integration_matrix=bool(mf_cfg.use_integration_matrix),
                seed=seed + i if seed is not None else None,
                ts_sigma_0=float(extra.get("ts_sigma_0", 0.30)),
                ts_sigma_min=float(extra.get("ts_sigma_min", 0.02)),
            )
        return MultiAgentPolicy(policies)
    elif policy_name.startswith("ptf"):
        # ptf_k{K} -- Probe-Then-Fit with probe_episodes=K
        # e.g. "ptf_k5", "ptf_k8", "ptf_k12"
        probe_ep = int(policy_name.split("k")[-1]) if "_k" in policy_name else 8
        mf_cfg = config.collaborative_filtering.matrix_factorization_cf
        return ProbeThenFitPolicy(
            num_agents=num_agents,
            num_targets=num_targets,
            latent_dim=mf_cfg.latent_dim,
            probe_episodes=probe_ep,
            ucb_c=ucb_c,
            learning_rate=mf_cfg.learning_rate,
            lambda_reg=mf_cfg.lambda_reg,
            epsilon=mf_cfg.epsilon,
            epsilon_decay=mf_cfg.epsilon_decay,
            epsilon_min=mf_cfg.epsilon_min,
            anti_signal_weight=mf_cfg.anti_signal_weight,
            use_integration_matrix=bool(mf_cfg.use_integration_matrix),
            seed=seed,
            allow_noop=allow_noop,
        )
    else:
        raise ValueError(f"Unknown policy: {policy_name!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=35)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override-json", default="{}")
    args = parser.parse_args()

    overrides = json.loads(args.override_json)
    extra = {}

    config = load_config(args.config)
    import dataclasses
    config = dataclasses.replace(config, seed=args.seed)
    config, extra = apply_overrides(config, overrides, extra)

    env, drones_config, targets_config, builder = build_env_and_configs(config, seed=args.seed)
    num_agents = env.num_drones
    num_targets = env.num_targets

    policy = build_policy(args.policy, config, num_agents, num_targets, extra)

    print(f"Sweep: {args.policy} seed={args.seed} episodes={args.episodes} overrides={overrides}")

    episode_metrics = []
    for ep in range(1, args.episodes + 1):
        ep_seed = args.seed + ep - 1
        m = run_one_episode(env, policy, ep, ep_seed)
        episode_metrics.append(m)
        policy.soft_reset()

    mgr = MetricsManager()
    summary = mgr.calc_total_episodes_metrics(episode_metrics)

    output = {
        "policy": args.policy,
        "seed": args.seed,
        "num_episodes": args.episodes,
        "overrides": overrides,
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

    print(f"Saved to {args.out}  avg_steps={summary.avg_steps:.1f}  success={summary.success_rate:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
