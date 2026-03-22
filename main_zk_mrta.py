"""
Demo script for ZK-MRTA environment with Random policy.

Demonstrates:
- Multi-agent PettingZoo environment setup
- Random policy baseline
- Episode execution
- Metrics collection and logging
"""

from typing import Dict, Any, List, Optional, Union, Tuple

import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tabula_drone.config import load_config
from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA, REWARD_MODE
from tabula_drone.logging import EnvironmentLogger
from tabula_drone.utils.engagement_analysis_utils import (
    build_target_x_drone_table,
    extract_drone_engagement_counts,
    load_analysis,
)
from tabula_drone.policies.random_policy import RandomPolicy
from tabula_drone.policies.min_ttk_oracle import OracleTimeToKillPolicy
from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
from tabula_drone.policies.ucb_cf_policy import UCBCFPolicy
from tabula_drone.policies.selfish_ep_greedy_cf_policy import SelfishEpGreedyCFPolicy
from tabula_drone.policies.coordinated_ep_greedy_cf_policy import CoordinatedEpGreedyCFPolicy
from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy
from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
from tabula_drone.policies.base import IPolicy, bind_diagnostics_provider
from tabula_drone.policies.utils.visualizer_bakery import enrich_learning_state_file
from tabula_drone.scenarios import ScenarioBuilder
from tabula_drone.utils.console_rendering import ConsolePrinter
from tabula_drone.utils.metrics_helper import calculate_derived_metrics, format_metric_display

CONFIG_PATH = "config/scenario.json"
printer = ConsolePrinter()


def get_env_diagnostics(env: DroneEngageZKMRTA) -> Dict[str, Any]:
    """Read the env-owned diagnostics payload used by the runner and logger."""
    if env.diagnostics is None:
        return {}
    return env.diagnostics.to_dict()


def show_learning_path(
    policy: Union[IPolicy, Dict[str, "SelfishEpGreedyCFPolicy"]],
    drones_config: List[Dict[str, Any]],
    targets_config: List[Dict[str, Any]],
) -> None:
    """Show current latent vectors for CF policy learning visualization."""
    # Handle MultiAgentPolicy wrapper
    from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
    if isinstance(policy, (dict, MultiAgentPolicy)):
        policies_dict = policy.policies if isinstance(policy, MultiAgentPolicy) else policy
        first_policy = next(iter(policies_dict.values()))
        latent_dim = first_policy.latent_dim
        # Collect agent vectors from each agent's private state
        agent_lv = [policies_dict[f"drone_{i}"].agent_lv for i in range(len(policies_dict))]
        # Use first agent's target estimates (they may differ between agents)
        target_lv = first_policy.target_lv
    else:
        cf_policy = policy
        latent_dim = cf_policy.latent_dim
        agent_lv = cf_policy.agent_lv
        target_lv = cf_policy.target_lv
    
    # Agent latent vectors
    agent_headers = ["Agent", "Weapon"] + [f"d{i}" for i in range(latent_dim)]
    agent_rows = []
    for i, drone_cfg in enumerate(drones_config):
        row = [f"A{i}", drone_cfg["weapon_type"][:4]]
        row.extend([f"{v:.3f}" for v in agent_lv[i]])
        agent_rows.append(row)
    agent_rows_sorted = sorted(agent_rows, key=lambda r: r[1])

    # Target latent vectors
    target_headers = ["Target", "Class"] + [f"d{i}" for i in range(latent_dim)]
    target_rows = []
    for i, target_cfg in enumerate(targets_config):
        row = [f"T{i}", target_cfg["class_type"][:4]]
        row.extend([f"{v:.3f}" for v in target_lv[i]])
        target_rows.append(row)
    target_rows_sorted = sorted(target_rows, key=lambda r: r[1])
    printer.learning_path(
        agent_headers=agent_headers,
        agent_rows=agent_rows_sorted,
        target_headers=target_headers,
        target_rows=target_rows_sorted,
    )


def create_policy(
    policy_type: str,
    config: Any,
    drones_config: List[Dict[str, Any]],
    num_targets: Optional[int] = None,
) -> IPolicy:
    """
    Factory function to create a policy instance.
    
    Args:
        policy_type: Type of policy ("min_ttk_oracle", "max_damage_oracle", "random", "ep_greedy_cf", "ucb_cf")
        config: ScenarioConfig with seed and policy settings
        drones_config: List of drone configurations for weapon profiles
        num_targets: Number of targets (required for collaborative_filtering)
    
    Returns:
        Policy instance
    """
    if policy_type == "min_ttk_oracle":
        agent_weapon_profiles = {
            f"drone_{idx}": dict(config.mappings.weapon_damage_profile_mapping[drone_cfg["weapon_type"]])
            for idx, drone_cfg in enumerate(drones_config)
        }
        return OracleTimeToKillPolicy(
            agent_weapon_profiles=agent_weapon_profiles,
            seed=config.seed,
            allow_noop=config.policy.allow_noop,
        )
    elif policy_type == "max_damage_oracle":
        agent_weapon_profiles = {
            f"drone_{idx}": dict(config.mappings.weapon_damage_profile_mapping[drone_cfg["weapon_type"]])
            for idx, drone_cfg in enumerate(drones_config)
        }
        return OptimalAssignmentOracle(
            agent_weapon_profiles=agent_weapon_profiles,
            seed=config.seed,
            allow_noop=config.policy.allow_noop,
        )
    elif policy_type == "ucb_cf":
        if num_targets is None:
            raise ValueError("num_targets is required for ucb_cf policy")
        # Extract hyperparameters from config, use defaults if not specified
        ucb_cfg = None
        if config.collaborative_filtering:
            ucb_cfg = config.collaborative_filtering.ucb_cf
        return UCBCFPolicy(
            num_agents=len(drones_config),
            num_targets=num_targets,
            latent_dim=ucb_cfg.latent_dim if ucb_cfg and ucb_cfg.latent_dim else 2,
            learning_rate=ucb_cfg.learning_rate if ucb_cfg and ucb_cfg.learning_rate else 0.01,
            ucb_c=ucb_cfg.ucb_c if ucb_cfg and ucb_cfg.ucb_c else 0.5,
            seed=config.seed,
        )
    elif policy_type == "selfish_ep_greedy_cf":
        if num_targets is None:
            raise ValueError("num_targets is required for selfish_ep_greedy_cf policy")
        # Extract hyperparameters from dedicated config section
        if not config.collaborative_filtering or not config.collaborative_filtering.selfish_ep_greedy_cf:
            raise ValueError("selfish_ep_greedy_cf policy requires collaborative_filtering.selfish_ep_greedy_cf config section")
        selfish_cfg = config.collaborative_filtering.selfish_ep_greedy_cf
        # Create one policy instance per agent (true decentralization)
        num_agents = len(drones_config)
        policies = {}
        for agent_idx in range(num_agents):
            agent_id = f"drone_{agent_idx}"
            policies[agent_id] = SelfishEpGreedyCFPolicy(
                num_targets=num_targets,
                agent_idx=agent_idx,
                num_agents=num_agents,
                latent_dim=selfish_cfg.latent_dim if selfish_cfg.latent_dim else 2,
                learning_rate=selfish_cfg.learning_rate if selfish_cfg.learning_rate else 0.01,
                epsilon=selfish_cfg.epsilon if selfish_cfg.epsilon else 0.3,
                epsilon_decay=selfish_cfg.epsilon_decay if selfish_cfg.epsilon_decay else 0.99,
                epsilon_min=selfish_cfg.epsilon_min if selfish_cfg.epsilon_min else 0.05,
                social_trust_factor=selfish_cfg.social_trust_factor if getattr(selfish_cfg, "social_trust_factor", None) is not None else 0.3,
                divergence_threshold=selfish_cfg.divergence_threshold if getattr(selfish_cfg, "divergence_threshold", None) is not None else 0.5,
                confidence_threshold=selfish_cfg.confidence_threshold if getattr(selfish_cfg, "confidence_threshold", None) is not None else 0.8,
                social_reward_clip_min=selfish_cfg.social_reward_clip_min if getattr(selfish_cfg, "social_reward_clip_min", None) is not None else -0.5,
                max_episodes=selfish_cfg.max_episodes if getattr(selfish_cfg, "max_episodes", None) is not None else 100,
                seed=config.seed + agent_idx if config.seed else None,
            )
        return MultiAgentPolicy(policies)
    elif policy_type == "coordinated_ep_greedy_cf":
        if num_targets is None:
            raise ValueError("num_targets is required for coordinated_ep_greedy_cf policy")
        # Extract hyperparameters from dedicated config section (required)
        if not config.collaborative_filtering or not config.collaborative_filtering.coordinated_ep_greedy_cf:
            raise ValueError("coordinated_ep_greedy_cf policy requires collaborative_filtering.coordinated_ep_greedy_cf config section")
        coord_cfg = config.collaborative_filtering.coordinated_ep_greedy_cf
        # Create one policy instance per agent (true decentralization with coordination)
        num_agents = len(drones_config)
        policies = {}
        for agent_idx in range(num_agents):
            agent_id = f"drone_{agent_idx}"
            policies[agent_id] = CoordinatedEpGreedyCFPolicy(
                num_targets=num_targets,
                agent_idx=agent_idx,
                num_agents=num_agents,
                latent_dim=coord_cfg.latent_dim if coord_cfg.latent_dim else 2,
                learning_rate=coord_cfg.learning_rate if coord_cfg.learning_rate else 0.01,
                epsilon=coord_cfg.epsilon if coord_cfg.epsilon else 0.3,
                epsilon_decay=coord_cfg.epsilon_decay if coord_cfg.epsilon_decay else 0.99,
                epsilon_min=coord_cfg.epsilon_min if coord_cfg.epsilon_min else 0.05,
                social_trust_factor=coord_cfg.social_trust_factor if getattr(coord_cfg, "social_trust_factor", None) is not None else 0.3,
                divergence_threshold=coord_cfg.divergence_threshold if getattr(coord_cfg, "divergence_threshold", None) is not None else 0.5,
                confidence_threshold=coord_cfg.confidence_threshold if getattr(coord_cfg, "confidence_threshold", None) is not None else 0.8,
                social_reward_clip_min=coord_cfg.social_reward_clip_min if getattr(coord_cfg, "social_reward_clip_min", None) is not None else -0.5,
                max_episodes=coord_cfg.max_episodes if getattr(coord_cfg, "max_episodes", None) is not None else 100,
                seed=config.seed + agent_idx if config.seed else None,
            )
        return MultiAgentPolicy(policies)
    elif policy_type == "matrix_factorization_cf":
        if num_targets is None:
            raise ValueError("num_targets is required for matrix_factorization_cf policy")
        # Extract hyperparameters from dedicated top-level config section
        mf_cfg = config.matrix_factorization_cf
        # Create one policy instance per agent (true decentralization)
        num_agents = len(drones_config)
        policies = {}
        for agent_idx in range(num_agents):
            agent_id = f"drone_{agent_idx}"
            policies[agent_id] = MatrixFactorizationPolicy(
                num_targets=num_targets,
                agent_idx=agent_idx,
                num_agents=num_agents,
                latent_dim=mf_cfg.latent_dim if mf_cfg and mf_cfg.latent_dim else 8,
                learning_rate=mf_cfg.learning_rate if mf_cfg and mf_cfg.learning_rate else 0.01,
                lambda_reg=mf_cfg.lambda_reg if mf_cfg and mf_cfg.lambda_reg is not None else 0.02,
                epsilon=mf_cfg.epsilon if mf_cfg and mf_cfg.epsilon is not None else 0.20,
                epsilon_decay=mf_cfg.epsilon_decay if mf_cfg and mf_cfg.epsilon_decay is not None else 1.0,
                epsilon_min=mf_cfg.epsilon_min if mf_cfg and mf_cfg.epsilon_min is not None else 0.02,
                anti_signal_weight=mf_cfg.anti_signal_weight if mf_cfg and mf_cfg.anti_signal_weight is not None else 0.1,
                selection_noise=mf_cfg.selection_noise if mf_cfg and hasattr(mf_cfg, 'selection_noise') and mf_cfg.selection_noise is not None else 0.0,
                seed=config.seed + agent_idx if config.seed else None,
            )
        return MultiAgentPolicy(policies)
    else:
        return RandomPolicy(seed=config.seed, allow_noop=config.policy.allow_noop)


def create_all_policies(
    config: Any,
    drones_config: List[Dict[str, Any]],
    num_targets: int,
) -> Dict[str, IPolicy]:
    """
    Create all policy instances upfront.
    
    Args:
        config: ScenarioConfig with seed and policy settings
        drones_config: List of drone configurations for weapon profiles
        num_targets: Number of targets in the environment
    
    Returns:
        Dict mapping policy_type to Policy instance
    """
    policies = {}
    for policy_type in config.policy.type:
        policies[policy_type] = create_policy(policy_type, config, drones_config, num_targets)
    return policies


def run_episode(
    env: DroneEngageZKMRTA,
    policy: IPolicy,
    episode_num: int,
    verbose: bool = False,
    environment_logger: Optional[EnvironmentLogger] = None,
    seed: Optional[int] = None,
    total_episodes: Optional[int] = None,
    flush_interval: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a single episode with the given policy.
    
    Args:
        env: ZK-MRTA environment
        policy: Policy for action selection
        episode_num: Episode number for logging
        verbose: If True, print step-by-step details
        environment_logger: Optional EnvironmentLogger for capturing episode data
        seed: Random seed used for this episode (for logger)
    
    Returns:
        Episode metrics dictionary
    """
    # Reset environment
    # ???
    # obs, info = env.reset()
    obs, infos = env.reset(seed=seed)
    shared_info = get_env_diagnostics(env)
    bind_diagnostics_provider(policy, lambda: get_env_diagnostics(env))
    
    if environment_logger:
        environment_logger.start_episode(
            env=env,
            reset_info=shared_info,
            seed=seed,
            episode_num=episode_num,
            total_episodes=total_episodes,
        )
    
    if verbose:
        printer.episode_start(
            episode_num=episode_num,
            num_drones=env.num_drones,
            num_targets=env.num_targets,
            target_classes=shared_info["target_classes"],
            weapon_types=shared_info["weapon_types"],
            target_hps=shared_info["target_hps"],
        )
    
    # Initialize tracking
    total_rewards = {agent_id: 0.0 for agent_id in env.agents}
    step_count = 0
    done = False
    overkill_events: List[Dict[int, float]] = []
    total_effective_damage = 0.0
    total_collisions = 0
    
    # Episode loop
    while not done:
        step_count += 1
        reference_agent_id = env.agents[0]
        
        # Policy selects actions for all agents (uniform interface)
        actions = policy.select_actions(obs, infos)

        # Environment step
        obs, rewards, terminations, truncations, infos = env.step(actions)
        shared_info = get_env_diagnostics(env)

        # Update policy
        policy.update(obs)

        # Check termination
        terminated = terminations[reference_agent_id]
        truncated = truncations[reference_agent_id]
        
        if environment_logger:
            environment_logger.log_step(
                step_num=step_count,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                info=shared_info,
            )
            
            # Periodic flush for continuous mode
            if flush_interval and step_count % flush_interval == 0:
                environment_logger.flush_episode(step_count)
                environment_logger.handle_flush(step_count)
        
        # Update total rewards and track effective damage
        for agent_id in env.agents:
            total_rewards[agent_id] += rewards[agent_id]
            
        # Sum absolute effective damage from ground truth (info)
        total_effective_damage += sum(shared_info["effective_damage"].values())
        
        # Track collisions
        total_collisions += shared_info.get("collisions", 0)
        
        # Track overkill
        if "overkill" in shared_info:
            overkill_events.append(shared_info["overkill"])
        
        # Verbose logging
        if verbose:
            printer.episode_step(
                step_count=step_count,
                actions=actions,
                target_hps=shared_info["target_hps"],
                target_active=shared_info["target_active"],
                rewards=rewards,
                overkill=shared_info.get("overkill"),
            )
        
        # Check termination
        done = terminated or truncated
    
    # Finalize logger (save is handled by caller for best-episode tracking)
    if environment_logger:
        environment_logger.end_episode(
            total_rewards=total_rewards,
            done_reason=shared_info.get("done_reason"),
        )
    
    # Compute final metrics
    targets_neutralized = shared_info.get("cumulative_neutralizations", sum(1 for active in shared_info['target_active'] if not active))
    total_ammo_used = sum(shared_info['ammo_used'].values())
    total_overkill = sum(
        sum(overkill.values()) for overkill in overkill_events
    )
    
    # Print summary
    if verbose:
        printer.episode_summary(
            episode_num=episode_num,
            step_count=step_count,
            total_rewards=total_rewards,
            ammo_used=shared_info["ammo_used"],
            weapon_types=shared_info["weapon_types"],
            done_reason=shared_info.get("done_reason", "N/A"),
            targets_neutralized=targets_neutralized,
            total_ammo_used=total_ammo_used,
        )
    
    # Compute potential damage from actual weapon profiles and ammo used per agent
    total_potential_damage = 0.0
    for agent_id, ammo in shared_info['ammo_used'].items():
        agent_idx = int(agent_id.split('_')[1])
        weapon_type = shared_info['weapon_types'][agent_idx]
        damage_per_shot = sum(env.weapon_damage_profile_mapping[weapon_type].values())
        total_potential_damage += ammo * damage_per_shot
    
    # Return metrics
    return {
        "episode": episode_num,
        "steps": step_count,
        "targets_neutralized": targets_neutralized,
        "total_ammo_used": total_ammo_used,
        "total_overkill": total_overkill,
        "done_reason": shared_info.get("done_reason"),
        "agent_rewards": total_rewards.copy(),
        "overkill_events": len(overkill_events),
        "total_effective_damage": total_effective_damage,
        "total_potential_damage": total_potential_damage,
        "total_collisions": total_collisions,
    }


def compute_alignment_score(
    agent_lv: np.ndarray,
    target_lv: np.ndarray,
    drones_config: List[Dict[str, Any]],
    targets_config: List[Dict[str, Any]],
) -> float:
    """
    Compute alignment score measuring how well latent vectors capture weapon-class relationships.
    
    Computes centroid of agent vectors per weapon type and target vectors per class type,
    then returns average dot product of optimal pairs: structural→A, breach→B, systems→C.
    
    Args:
        agent_lv: Agent latent vectors, shape (num_agents, latent_dim)
        target_lv: Target latent vectors, shape (num_targets, latent_dim)
        drones_config: List of drone configs with 'weapon_type' key
        targets_config: List of target configs with 'class_type' key
    
    Returns:
        Alignment score in [-1, 1], where 1 = perfect alignment
    """
    optimal_pairs = {
        "structural": "A",
        "breach": "B",
        "systems": "C",
    }
    
    # Compute agent centroids per weapon type
    agent_centroids = {}
    for weapon in optimal_pairs.keys():
        indices = [i for i, d in enumerate(drones_config) if d["weapon_type"] == weapon]
        if indices:
            agent_centroids[weapon] = np.mean(agent_lv[indices], axis=0)
    
    # Compute target centroids per class type
    target_centroids = {}
    for cls in optimal_pairs.values():
        indices = [i for i, t in enumerate(targets_config) if t["class_type"] == cls]
        if indices:
            target_centroids[cls] = np.mean(target_lv[indices], axis=0)
    
    # Compute average dot product of optimal pairs
    dot_products = []
    for weapon, cls in optimal_pairs.items():
        if weapon in agent_centroids and cls in target_centroids:
            dot = np.dot(agent_centroids[weapon], target_centroids[cls])
            dot_products.append(dot)
    
    if not dot_products:
        return 0.0
    
    return float(np.mean(dot_products))


def analyze_agent_clustering(policy, drone_weapon_map):
    """
    Prints similarity scores between drones to see if they are clustering by weapon.
    drone_weapon_map: List of weapons assigned to agents (e.g., ['structural', 'breach', ...])
    """
    vectors = policy.agent_lv
    sim_matrix = cosine_similarity(vectors)
    similarity_lines = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            w1 = drone_weapon_map[i]
            w2 = drone_weapon_map[j]
            similarity = sim_matrix[i, j]
            
            # Highlight if drones with the SAME weapon are becoming similar
            match_status = "[MATCH]" if w1 == w2 else "       "
            similarity_lines.append(
                f"Agent {i}({w1[:4]}) vs Agent {j}({w2[:4]}): {similarity:.4f} {match_status}"
            )
    printer.agent_clustering(drone_weapon_map, similarity_lines)


def show_policy_performance_summary(
    config: Any,
    all_metrics: List[Dict[str, Any]],
) -> None:
    table_data = []
    for policy_type in config.policy.type:
        policy_metrics = [m for m in all_metrics if m["policy_type"] == policy_type]
        if policy_metrics:
            n = len(policy_metrics)
            avg_steps = sum(m["steps"] for m in policy_metrics) / n
            avg_targets = sum(m["targets_neutralized"] for m in policy_metrics) / n
            avg_ammo = sum(m["total_ammo_used"] for m in policy_metrics) / n
            avg_overkill = sum(m["total_overkill"] for m in policy_metrics) / n
            avg_reward = sum(sum(m["agent_rewards"].values()) for m in policy_metrics) / n
            success_count = sum(1 for m in policy_metrics if m["done_reason"] == "all_targets_neutralized")
            success_rate = (success_count / n) * 100
            ammo_eff = avg_targets / avg_ammo if avg_ammo > 0 else 0.0
            avg_eff_dmg = sum(m["total_effective_damage"] for m in policy_metrics) / n
            avg_pot_dmg = sum(m["total_potential_damage"] for m in policy_metrics) / n
            dmg_eff = avg_eff_dmg / avg_pot_dmg if avg_pot_dmg > 0 else 0.0
            table_data.append([
                policy_type,
                avg_steps,
                f"{avg_targets:.1f}",
                f"{avg_ammo:.1f}",
                f"{avg_overkill:.1f}",
                f"{avg_reward:.1f}",
                f"{success_rate:.0f}%",
                f"{ammo_eff:.3f}",
                f"{dmg_eff:.1%}",
            ])

    table_data.sort(key=lambda row: row[1])
    for row in table_data:
        row[1] = f"{row[1]:.1f}"

    headers = ["Policy", "Avg Steps", "Avg Targets", "Avg Ammo", "Avg Overkill", "Avg Reward", "Success %", "Ammo Eff", "Dmg Eff"]
    printer.policy_performance_summary(table_data, headers)


def show_policy_best_episode_performance_vs_random(
    config: Any,
    all_metrics: List[Dict[str, Any]],
) -> None:
    mode = config.environment.mode
    policy_best_metrics = {}
    for policy_type in config.policy.type:
        policy_metrics = [m for m in all_metrics if m["policy_type"] == policy_type]
        if policy_metrics:
            # For continuous mode, we use the final state; for episodic, we find the best (shortest) episode
            if mode == "continuous":
                best_ep = policy_metrics[-1]
            else:
                best_ep = min(policy_metrics, key=lambda m: m["steps"])
            
            policy_best_metrics[policy_type] = calculate_derived_metrics(best_ep, mode=mode)

    if "random" in policy_best_metrics and len(policy_best_metrics) > 1:
        rand = policy_best_metrics["random"]

        def fmt_pct(val, base, fmt, higher_better=True, is_str=False):
            if is_str:
                return val
            if base == 0:
                return fmt.format(val)
            pct = ((val - base) / base) * 100
            if not higher_better:
                pct = -pct
            sign = "+" if pct > 0 else ""
            return f"{fmt.format(val)} ({sign}{pct:.0f}%)"

        cmp_data = []
        if mode == "continuous":
            headers = ["Policy", "Throughput (N/100)", "Coordination (N/C)", "Ammo Eff", "Dmg Eff"]
            for pt in config.policy.type:
                if pt in policy_best_metrics:
                    m = policy_best_metrics[pt]
                    if pt == "random":
                        cmp_data.append([
                            pt,
                            f"{m['throughput']:.1f} (base)",
                            f"{m['coordination_str']} (base)",
                            f"{m['ammo_eff']:.3f} (base)",
                            f"{m['dmg_eff']:.1%} (base)",
                        ])
                    else:
                        cmp_data.append([
                            pt,
                            fmt_pct(m["throughput"], rand["throughput"], "{:.1f}", True),
                            fmt_pct(m["coordination_score"], rand["coordination_score"], "{:.2f}", True) if m["coordination_score"] != float('inf') else "Perfect",
                            fmt_pct(m["ammo_eff"], rand["ammo_eff"], "{:.3f}", True),
                            fmt_pct(m["dmg_eff"], rand["dmg_eff"], "{:.1%}", True),
                        ])
        else:
            headers = ["Policy", "Best Steps", "Shots/Target", "Ammo Eff", "Dmg Eff"]
            for pt in config.policy.type:
                if pt in policy_best_metrics:
                    m = policy_best_metrics[pt]
                    if pt == "random":
                        cmp_data.append([
                            pt,
                            f"{m['best_steps']} (base)",
                            f"{m['shots_per_target']:.1f} (base)",
                            f"{m['ammo_eff']:.3f} (base)",
                            f"{m['dmg_eff']:.1%} (base)",
                        ])
                    else:
                        cmp_data.append([
                            pt,
                            fmt_pct(m["best_steps"], rand["best_steps"], "{}", False),
                            fmt_pct(m["shots_per_target"], rand["shots_per_target"], "{:.1f}", False),
                            fmt_pct(m["ammo_eff"], rand["ammo_eff"], "{:.3f}", True),
                            fmt_pct(m["dmg_eff"], rand["dmg_eff"], "{:.1%}", True),
                        ])

        printer.policy_performance_comparison(
            mode=mode,
            mappings_file=config.mappings_file,
            headers=headers,
            cmp_data=cmp_data,
        )


def show_engagement_tables(
    engagement_tables: Dict[str, Any],
) -> None:
    print_order = ["random", "max_damage_oracle", "selfish_ep_greedy_cf"]
    for pt in print_order:
        if pt not in engagement_tables:
            continue
        headers, payload = engagement_tables[pt]
        printer.engagement_table(pt, headers, payload)


def main():
    """Main demo execution."""
    
    # Load configuration from file
    config = load_config(CONFIG_PATH)
    
    # Environment configuration using ScenarioBuilder
    builder = ScenarioBuilder(
        world_size=config.world.size,
        seed=config.seed,
        class_attribute_mapping=config.mappings.class_attribute_mapping,
        weapon_damage_profile_mapping=config.mappings.weapon_damage_profile_mapping,
    )
    
    # Configure drones with count, region, and weapon distribution
    builder.with_drones(
        count=config.drones.count,
        region=config.drones.region,
        min_distance_between_drones=config.drones.min_distance_between_drones,
        weapon_distribution=config.drones.weapon_distribution
    )
    
    # Configure targets with spatial constraints and class distribution
    builder.with_targets(
        count=config.targets.count,
        region=config.targets.region,
        class_distribution=config.targets.class_distribution,
        min_distance_from_drones=config.targets.min_distance_from_drones,
        min_distance_between_targets=config.targets.min_distance_between_targets
    )
    
    # Build configurations
    drones_config, targets_config = builder.build()
    
    # Get CF noise parameters from config (with defaults)
    cf_config = getattr(config, 'collaborative_filtering', None)
    reward_noise = cf_config.reward_noise if cf_config else 0.1
    observation_noise = cf_config.observation_noise if cf_config else 0.05
    
    # Run episodes
    if config.environment.mode == "episodic":
        num_episodes = config.environment.episodic.num_episodes
    else:
        # Continuous mode is effectively 1 long episode per policy
        num_episodes = 1
    all_metrics = []
    
    printer.demo_header(
        config_path=CONFIG_PATH,
        mappings_file=config.mappings_file,
        world_size=config.world.size,
        seed=config.seed,
        policy_types=config.policy.type,
        num_episodes=num_episodes,
    )
    printer.initial_setup(
        class_attribute_mapping=config.mappings.class_attribute_mapping,
        weapon_damage_profile_mapping=config.mappings.weapon_damage_profile_mapping,
        drones_config=drones_config,
        targets_config=targets_config,
    )
    printer.optimal_engagement_prediction(
        drones_config,
        targets_config,
        config.mappings.class_attribute_mapping,
        config.mappings.weapon_damage_profile_mapping,
    )
    printer.separator()
    
    # Create EnvironmentLogger for structured logging orchestration
    environment_logger = EnvironmentLogger(
        output_dir=config.logging.output_dir,
        scenario_id=config.environment.scenario_id,
        mode=config.environment.mode
    )
    
    # Determine noise settings based on active policies
    mf_config = getattr(config, 'matrix_factorization_cf', None)
    cf_config = getattr(config, 'collaborative_filtering', None)
    
    # Priority: Dedicated noise settings in the specific policy section
    # Default to 0.0 if neither is provided.
    if mf_config and (mf_config.reward_noise is not None or mf_config.observation_noise is not None):
        reward_noise = mf_config.reward_noise if mf_config.reward_noise is not None else 0.05
        observation_noise = mf_config.observation_noise if mf_config.observation_noise is not None else 0.05
    elif cf_config:
        reward_noise = cf_config.reward_noise if cf_config.reward_noise is not None else 0.1
        observation_noise = cf_config.observation_noise if cf_config.observation_noise is not None else 0.05
    else:
        reward_noise = 0.0
        observation_noise = 0.0
    
    # Create single environment (reused across all policies)
    num_targets = len(targets_config)
    env = DroneEngageZKMRTA(
        world_size=config.world.size,
        max_steps=config.environment.max_steps,
        drones_config=drones_config,
        targets_config=targets_config,
        scenario_id=config.environment.scenario_id,
        class_attribute_mapping=config.mappings.class_attribute_mapping,
        weapon_damage_profile_mapping=config.mappings.weapon_damage_profile_mapping,
        reward_noise=reward_noise,
        observation_noise=observation_noise,
        mode=config.environment.mode,
        builder=builder,
    )
    
    # Create all policies upfront
    policies = create_all_policies(config, drones_config, num_targets)
    
    # Run each policy
    engagement_tables = {}
    for policy_type, policy in policies.items():
        printer.policy_run_header(policy_type)
        
        # Get policy metadata from policy attributes
        is_deterministic = policy.is_deterministic
        effective_episodes = 1 if is_deterministic else num_episodes
        
        # Start policy run in EnvironmentLogger
        environment_logger.start_policy(
            policy_type,
            is_deterministic=is_deterministic,
        )
        
        for episode_num in range(1, effective_episodes + 1):
            # Soft reset CF policy for new episode (preserves agent latent vectors)
            if not is_deterministic and episode_num > 1:
                policy.soft_reset()
            
            pre_episode_state = None
            post_episode_state = None
            
            # Snapshot latent vectors BEFORE episode (for correct learning_path capture)
            if not is_deterministic:
                # For CF policies: use get_learning_state() for state capture
                pre_episode_state = policy.get_learning_state()
                # Extract agent_lv and target_lv for alignment score computation
                agent_lv = np.array([agent["agent_lv"] for agent in pre_episode_state["agents"]])
                target_lv = np.array(pre_episode_state["agents"][0]["target_lv"])
                pre_episode_lv = (agent_lv, target_lv)
            else:
                pre_episode_lv = None
            
            # Prepare entities metadata for learning state logging
            entities = None
            if not is_deterministic and (policy_type == "selfish_ep_greedy_cf" or policy_type == "matrix_factorization_cf"):
                entities = {
                    "agents": [
                        {
                            "agent_idx": i,
                            "agent_id": f"drone_{i}",
                            "weapon_type": drones_config[i]["weapon_type"],
                            "weapon_damage_profile": dict(
                                config.mappings.weapon_damage_profile_mapping[
                                    drones_config[i]["weapon_type"]
                                ]
                            ),
                        }
                        for i in range(len(drones_config))
                    ],
                    "targets": [
                        {
                            "target_idx": j,
                            "target_id": f"target_{j}",
                            "class_type": targets_config[j]["class_type"],
                            "class_attributes": dict(
                                config.mappings.class_attribute_mapping[
                                    targets_config[j]["class_type"]
                                ]
                            ),
                        }
                        for j in range(len(targets_config))
                    ],
                }
            # This ensures that ENV noise/ordering is reproducible across runs
            episode_seed = config.seed + episode_num if config.seed is not None else None

            # Determine flush interval for continuous mode
            flush_interval = None
            if config.environment.mode == "continuous":
                flush_interval = config.environment.continuous.logging_interval_steps

            environment_logger.configure_continuous_flush(
                episode_num=episode_num,
                learning_state_provider=(
                    policy.get_learning_state if not is_deterministic else None
                ),
                num_agents=getattr(policy, "num_agents", len(drones_config)),
                num_targets=getattr(policy, "num_targets", len(targets_config)),
                latent_dim=getattr(policy, "latent_dim", None),
                entities=entities,
            )

            metrics = run_episode(
                env=env,
                policy=policy,
                episode_num=episode_num,
                verbose=config.environment.verbose,
                environment_logger=environment_logger,
                seed=episode_seed,  #config.seed
                total_episodes=num_episodes,
                flush_interval=flush_interval,
            )
            metrics["policy_type"] = policy_type
            all_metrics.append(metrics)
            
            # Capture post-episode state for CF policies
            if not is_deterministic:
                post_episode_state = policy.get_learning_state()
            
            # Compute alignment score for CF policies (used by both trackers)
            # Save learning state for every episode
            if not is_deterministic:
                environment_logger.save_learning_state(
                    pre_state=pre_episode_state,
                    post_state=post_episode_state,
                    episode_num=episode_num,
                    num_agents=getattr(policy, "num_agents", len(drones_config)),
                    num_targets=getattr(policy, "num_targets", len(targets_config)),
                    latent_dim=getattr(policy, "latent_dim", None),
                    entities=entities,
                )
            
            # Persist per-episode outputs through EnvironmentLogger
            environment_logger.persist_episode_outputs(
                episode_num=episode_num,
                steps=metrics["steps"],
            )
            
            # Per-run summary
            if config.environment.mode == "continuous":
                derived = calculate_derived_metrics(metrics, mode="continuous")
                printer.continuous_run_progress(
                    steps=metrics["steps"],
                    throughput=derived["throughput"],
                    coordination=derived["coordination_str"],
                    ammo_eff=derived["ammo_eff"],
                    collisions=metrics["total_collisions"],
                )
            else:
                total_reward = sum(metrics["agent_rewards"].values())
                printer.episode_run_progress(
                    episode_num=episode_num,
                    steps=metrics["steps"],
                    targets_neutralized=metrics["targets_neutralized"],
                    total_effective_damage=metrics["total_effective_damage"],
                    total_overkill=metrics["total_overkill"],
                    total_reward=total_reward,
                )
            
            # Print learning path for CF policies
            if (
                not is_deterministic
                and config.environment.verbose
                and callable(getattr(policy, "get_learning_state", None))
            ):
                show_learning_path(policy, drones_config, targets_config)
            
            # Debug: Analyze agent clustering for CF policies
            # if False and not is_deterministic:
            #     drone_weapons = [d["weapon_type"] for d in drones_config]
            #     analyze_agent_clustering(policy, drone_weapons)
        
        # Finalize policy run - saves selected episodes (first/best/mid or only)
        result = environment_logger.finalize_policy()
        if config.environment.mode != "continuous":
            printer.saved_episodes(result["files"])
        steps = result['steps']
        
        # Enrich learning state for milestone episodes with t-SNE (offline post-processing)
        # This keeps the training loop fast while providing rich visualizations for key episodes.
        if not is_deterministic and policy_type == "matrix_factorization_cf":
            milestones = result.get('milestones', {})
            milestone_episodes = list(set(milestones.values())) # unique episode numbers
            for category, ep_num in milestones.items():
                if environment_logger.mode == "continuous":
                    filename = "learning_state_continuous_final.json"
                else:
                    filename = f"learning_state_ep{ep_num:02d}.json"
                    
                    state_file = os.path.join(
                        environment_logger.get_learning_state_dir(),
                        filename
                    )
                    if os.path.exists(state_file):
                        enrich_learning_state_file(state_file)

        if 'best' in steps:
            printer.policy_steps_summary(
                first=steps["first"],
                best=steps["best"],
                mid=steps["mid"],
            )
        elif 'final' in steps:
            # For continuous mode, show cumulative results at the end of the policy run
            printer.policy_final_summary(
                steps=metrics["steps"],
                targets_neutralized=metrics["targets_neutralized"],
                total_effective_damage=metrics["total_effective_damage"],
                total_overkill=metrics["total_overkill"],
                total_collisions=metrics["total_collisions"],
            )
        else:
            printer.policy_first_step(steps.get("first", "N/A"))

        best_episode_num = result.get("best_episode_num")
        if best_episode_num is not None:
            # Check for analysis file (naming depends on mode)
            if config.environment.mode == "continuous":
                analysis_path = os.path.join(
                    environment_logger.get_analysis_dir(),
                    "analysis_continuous_final.json",
                )
            else:
                analysis_path = os.path.join(
                    environment_logger.get_analysis_dir(),
                    f"analysis_ep{best_episode_num:02d}.json",
                )
            if os.path.exists(analysis_path):
                analysis_data = load_analysis(analysis_path)
                drone_engagement_counts = extract_drone_engagement_counts(analysis_data)
                engagement_headers, engagement_rows = build_target_x_drone_table(
                    drone_engagement_counts
                )
                engagement_tables[policy_type] = (engagement_headers, engagement_rows)
            else:
                engagement_tables[policy_type] = (None, f"Analysis file not found: {analysis_path}")

    if config.environment.verbose:
        # Aggregate statistics across all episodes (all policies)
        total_episodes = len(all_metrics)
        avg_steps = sum(m["steps"] for m in all_metrics) / total_episodes
        avg_targets = sum(m["targets_neutralized"] for m in all_metrics) / total_episodes
        avg_ammo = sum(m["total_ammo_used"] for m in all_metrics) / total_episodes
        avg_overkill = sum(m["total_overkill"] for m in all_metrics) / total_episodes
        per_policy_rows = []
        # Per-policy statistics
        for policy_type in config.policy.type:
            policy_metrics = [m for m in all_metrics if m["policy_type"] == policy_type]
            if policy_metrics:
                per_policy_rows.extend([
                    f"\n  Policy '{policy_type}':",
                    f"    Avg Steps: {sum(m['steps'] for m in policy_metrics) / len(policy_metrics):.1f}",
                    f"    Avg Targets: {sum(m['targets_neutralized'] for m in policy_metrics) / len(policy_metrics):.1f}",
                ])
        per_agent_rows = []
        # Per-agent statistics (use last env for agent list)
        for agent_id in env.agents:
            avg_reward = sum(m["agent_rewards"][agent_id] for m in all_metrics) / total_episodes
            per_agent_rows.append(f"  {agent_id}: {avg_reward:.2f}")
        printer.aggregate_statistics(
            total_episodes=total_episodes,
            num_episodes=num_episodes,
            policy_count=len(config.policy.type),
            avg_steps=avg_steps,
            avg_targets=avg_targets,
            avg_ammo=avg_ammo,
            avg_overkill=avg_overkill,
            per_policy_rows=per_policy_rows,
            per_agent_rows=per_agent_rows,
        )
    
    # show_policy_performance_summary(config, all_metrics)

    show_policy_best_episode_performance_vs_random(config, all_metrics)

    # show_engagement_tables(engagement_tables)

    printer.demo_complete()


if __name__ == "__main__":
    main()
