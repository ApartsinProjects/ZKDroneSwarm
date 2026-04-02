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
from tabula_drone.envs.drone_engage_latent_mrta import DroneEngageLatentMRTA
from tabula_drone.logging import EnvironmentLogger
from tabula_drone.utils.engagement_analysis_utils import (
    build_target_x_drone_table,
    extract_drone_engagement_counts,
    load_analysis,
)
from tabula_drone.policies.random_policy import RandomPolicy
from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
from tabula_drone.policies.matrix_factorization_policy import MatrixFactorizationPolicy
from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
from tabula_drone.policies.base import IPolicy, bind_diagnostics_provider
from tabula_drone.policies.utils.visualizer_bakery import (
    TSNE_MODE_PER_EPISODE,
    TSNE_MODE_PER_EPISODE_ALIGNED,
    enrich_learning_state_dir,
    _iter_learning_state_files,
)
from tabula_drone.scenarios.latent_scenario_builder import LatentScenarioBuilder
from tabula_drone.utils.console_rendering import ConsolePrinter
from tabula_drone.utils.metrics_manager import (
    EpisodeMetrics,
    MetricsManager,
    PolicyRunSummary,
)

CONFIG_PATH = "config/scenario.json"
printer = ConsolePrinter()


def get_env_diagnostics(env: DroneEngageLatentMRTA) -> Dict[str, Any]:
    """Read the env-owned diagnostics payload used by the runner and logger."""
    if env.diagnostics is None:
        return {}
    return env.diagnostics.to_dict()


def attach_target_classes_to_learning_state(
    episode_state: Optional[Dict[str, Any]],
    env: DroneEngageLatentMRTA,
) -> Optional[Dict[str, Any]]:
    """Add shared target classes to a learning-state payload using env target order."""
    if episode_state is None:
        return None

    diagnostics = get_env_diagnostics(env)
    if isinstance(diagnostics.get("target_classes"), list):
        target_classes = list(diagnostics["target_classes"])
    else:
        targets = env.targets or []
        target_classes = [
            getattr(target, "class_type", f"mode_{getattr(target, 'mode_id', '?')}")
            for target in targets
        ]

    learning_state = dict(episode_state)
    learning_state["target_classes"] = target_classes
    return learning_state


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
        agent_emb = [policies_dict[f"drone_{i}"].agent_emb for i in range(len(policies_dict))]
        # Use first agent's target estimates (they may differ between agents)
        target_emb = first_policy.target_emb
    else:
        cf_policy = policy
        latent_dim = cf_policy.latent_dim
        agent_emb = cf_policy.agent_emb
        target_emb = cf_policy.target_emb
    
    # Agent latent vectors
    agent_headers = ["Agent", "Weapon"] + [f"d{i}" for i in range(latent_dim)]
    agent_rows = []
    for i, drone_cfg in enumerate(drones_config):
        row = [f"A{i}", drone_cfg["weapon_type"][:4]]
        row.extend([f"{v:.3f}" for v in agent_emb[i]])
        agent_rows.append(row)
    agent_rows_sorted = sorted(agent_rows, key=lambda r: r[1])

    # Target latent vectors
    target_headers = ["Target", "Class"] + [f"d{i}" for i in range(latent_dim)]
    target_rows = []
    for i, target_cfg in enumerate(targets_config):
        row = [f"T{i}", target_cfg["class_type"][:4]]
        row.extend([f"{v:.3f}" for v in target_emb[i]])
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
    if policy_type == "max_damage_oracle":
        return OptimalAssignmentOracle(
            seed=config.seed,
            allow_noop=config.policy.allow_noop,
        )
    elif policy_type == "matrix_factorization_cf":
        if num_targets is None:
            raise ValueError("num_targets is required for matrix_factorization_cf policy")
        # Extract hyperparameters from collaborative_filtering.matrix_factorization_cf section
        mf_cfg = config.collaborative_filtering.matrix_factorization_cf if config.collaborative_filtering else None
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
    env: DroneEngageLatentMRTA,
    policy: IPolicy,
    episode_num: int,
    verbose: bool = False,
    environment_logger: Optional[EnvironmentLogger] = None,
    seed: Optional[int] = None,
    total_episodes: Optional[int] = None,
    flush_interval: Optional[int] = None,
) -> EpisodeMetrics:
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
        Typed episode metrics
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
    total_net_damage = 0.0
    total_gross_damage = 0.0
    total_collisions = 0
    
    # Episode loop
    while not done:
        step_count += 1
        reference_agent_id = env.agents[0]
        
        # Policy selects actions for all agents (uniform interface)
        # Try passing env for oracle policies, fallback to basic signature for others
        try:
            actions = policy.select_actions(obs, infos, env=env)
        except TypeError:
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
        
        # Update total rewards and track net damage
        for agent_id in env.agents:
            total_rewards[agent_id] += rewards[agent_id]
            
        # Sum actual HP removed from targets from ground truth (info)
        total_net_damage += shared_info["net_damage"]
        total_gross_damage += shared_info["total_gross_damage"]
        
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
    
    # Extract fields from final diagnostics
    done_reason = shared_info.get("done_reason")
    targets_neutralized = shared_info.get("cumulative_neutralizations", 0)
    ammo_used_dict = shared_info.get("ammo_used", {})
    total_ammo_used = sum(ammo_used_dict.values())
    total_overkill = sum(sum(event.values()) for event in overkill_events)
    
    # Construct EpisodeMetrics directly (calculates efficiency in __post_init__)
    metrics = EpisodeMetrics(
        episode=episode_num,
        steps=step_count,
        mode=env.mode,
        done_reason=done_reason,
        targets_neutralized=targets_neutralized,
        total_ammo_used=total_ammo_used,
        total_overkill=total_overkill,
        total_net_damage=total_net_damage,
        total_gross_damage=total_gross_damage,
        total_collisions=total_collisions,
        agent_rewards=total_rewards.copy(),
        weapon_damage_profile_mapping=env.weapon_damage_profile_mapping,
    )

    # Print summary
    if verbose:
        printer.episode_summary(
            episode_num=episode_num,
            step_count=step_count,
            total_rewards=total_rewards,
            ammo_used=shared_info["ammo_used"],
            weapon_types=shared_info["weapon_types"],
            done_reason=metrics.done_reason or "N/A",
            targets_neutralized=metrics.targets_neutralized,
            total_ammo_used=metrics.total_ammo_used,
        )

    return metrics


def compute_alignment_score(
    agent_emb: np.ndarray,
    target_emb: np.ndarray,
    drones_config: List[Dict[str, Any]],
    targets_config: List[Dict[str, Any]],
) -> float:
    """
    Compute alignment score measuring how well latent vectors capture weapon-class relationships.
    
    Computes centroid of agent vectors per weapon type and target vectors per class type,
    then returns average dot product of optimal pairs: structural→A, breach→B, systems→C.
    
    Args:
        agent_emb: Agent embedding vectors, shape (num_agents, latent_dim)
        target_emb: Target embedding vectors, shape (num_targets, latent_dim)
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
            agent_centroids[weapon] = np.mean(agent_emb[indices], axis=0)
    
    # Compute target centroids per class type
    target_centroids = {}
    for cls in optimal_pairs.values():
        indices = [i for i, t in enumerate(targets_config) if t["class_type"] == cls]
        if indices:
            target_centroids[cls] = np.mean(target_emb[indices], axis=0)
    
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
    vectors = policy.agent_emb
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
    policy_summaries: Dict[str, PolicyRunSummary],
) -> None:
    table_data = []
    for policy_type in config.policy.type:
        policy_summary = policy_summaries.get(policy_type)
        if policy_summary:
            table_data.append([
                policy_type,
                policy_summary.avg_steps,
                f"{policy_summary.avg_targets:.1f}",
                f"{policy_summary.avg_ammo:.1f}",
                f"{policy_summary.avg_overkill:.1f}",
                f"{policy_summary.avg_reward:.1f}",
                f"{policy_summary.success_rate:.0f}%",
                f"{policy_summary.ammo_eff:.3f}",
                f"{policy_summary.dmg_eff:.1%}",
            ])

    table_data.sort(key=lambda row: row[1])
    for row in table_data:
        row[1] = f"{row[1]:.1f}"

    headers = ["Policy", "Avg Steps", "Avg Targets", "Avg Ammo", "Avg Overkill", "Avg Reward", "Success %", "Ammo Eff", "Dmg Eff"]
    printer.policy_performance_summary(table_data, headers)


def show_policy_best_episode_performance_vs_random(
    config: Any,
    policy_summaries: Dict[str, PolicyRunSummary],
) -> None:
    mode = config.environment.mode
    policy_best_metrics = {}
    for policy_type in config.policy.type:
        policy_summary = policy_summaries.get(policy_type)
        if policy_summary and policy_summary.representative_episode is not None:
            policy_best_metrics[policy_type] = policy_summary.representative_episode

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
                            f"{m.throughput:.1f} (base)",
                            f"{m.coordination_str} (base)",
                            f"{m.ammo_eff:.3f} (base)",
                            f"{m.dmg_eff:.1%} (base)",
                        ])
                    else:
                        cmp_data.append([
                            pt,
                            fmt_pct(m.throughput, rand.throughput, "{:.1f}", True),
                            fmt_pct(m.coordination_score, rand.coordination_score, "{:.2f}", True) if m.coordination_score != float('inf') else "Perfect",
                            fmt_pct(m.ammo_eff, rand.ammo_eff, "{:.3f}", True),
                            fmt_pct(m.dmg_eff, rand.dmg_eff, "{:.1%}", True),
                        ])
        else:
            headers = ["Policy", "Best Steps", "Shots/Target", "Ammo Eff", "Dmg Eff"]
            for pt in config.policy.type:
                if pt in policy_best_metrics:
                    m = policy_best_metrics[pt]
                    if pt == "random":
                        cmp_data.append([
                            pt,
                            f"{m.best_steps} (base)",
                            f"{m.shots_per_target:.1f} (base)",
                            f"{m.ammo_eff:.3f} (base)",
                            f"{m.dmg_eff:.1%} (base)",
                        ])
                    else:
                        cmp_data.append([
                            pt,
                            fmt_pct(m.best_steps, rand.best_steps, "{}", False),
                            fmt_pct(m.shots_per_target, rand.shots_per_target, "{:.1f}", False),
                            fmt_pct(m.ammo_eff, rand.ammo_eff, "{:.3f}", True),
                            fmt_pct(m.dmg_eff, rand.dmg_eff, "{:.1%}", True),
                        ])

        printer.policy_performance_comparison(
            mode=mode,
            mappings_file=None,
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

    if config.latent_world is None:
        raise ValueError("latent world_model requires parsed latent_world config")

    builder = LatentScenarioBuilder(
        world_size=config.world.size,
        latent_dim=config.latent_world.latent_dim,
        num_modes=config.latent_world.num_modes,
        drone_variance=config.latent_world.drone_variance,
        target_variance=config.latent_world.target_variance,
        seed=config.seed,
        center_mode=config.latent_world.center_mode,
    )
    builder.with_drones(
        count=config.drones.count,
        region=config.drones.region,
        min_distance_between_drones=config.drones.min_distance_between_drones,
    )
    builder.with_targets(
        count=config.targets.count,
        region=config.targets.region,
        min_distance_from_drones=config.targets.min_distance_from_drones,
        min_distance_between_targets=config.targets.min_distance_between_targets,
    )
    
    # Build configurations
    drones_config, targets_config = builder.build()
    
    # Run episodes
    if config.environment.mode == "episodic":
        num_episodes = config.environment.episodic.num_episodes
    else:
        # Continuous mode is effectively 1 long episode per policy
        num_episodes = 1
    all_metrics: List[EpisodeMetrics] = []
    policy_summaries: Dict[str, PolicyRunSummary] = {}
    
    printer.demo_header(
        config_path=CONFIG_PATH,
        mappings_file=None,
        world_size=config.world.size,
        seed=config.seed,
        policy_types=config.policy.type,
        num_episodes=num_episodes,
    )
    printer.separator()

    printer.latent_world_debug(drones_config, targets_config, precision=3, max_components=6)

    
    # Create EnvironmentLogger for structured logging orchestration
    environment_logger = EnvironmentLogger(
        output_dir=config.logging.output_dir,
        scenario_id=config.environment.scenario_id,
        mode=config.environment.mode
    )
    
    # Determine noise settings based on active policies
    mf_config = config.collaborative_filtering.matrix_factorization_cf if config.collaborative_filtering else None
    cf_config = config.collaborative_filtering
    
    # Priority: Use MF-specific noise if provided, otherwise fall back to CF-level noise
    if mf_config and mf_config.reward_noise is not None:
        reward_noise = mf_config.reward_noise
    elif cf_config and cf_config.reward_noise is not None:
        reward_noise = cf_config.reward_noise
    else:
        reward_noise = 0.0
    
    # Create single environment (reused across all policies)
    num_targets = len(targets_config)
    env = DroneEngageLatentMRTA(
        world_size=config.world.size,
        max_steps=config.environment.max_steps,
        drones_config=drones_config,
        targets_config=targets_config,
        scenario_id=config.environment.scenario_id,
        reward_noise=reward_noise,
        mode=config.environment.mode,
        builder=builder,
        latent_world={
            "latent_dim": config.latent_world.latent_dim,
            "num_modes": config.latent_world.num_modes,
            "drone_variance": config.latent_world.drone_variance,
            "target_variance": config.latent_world.target_variance,
            "target_hp": config.latent_world.target_hp,
            "center_mode": config.latent_world.center_mode,
        },
    )
        
    # Create all policies upfront
    policies = create_all_policies(config, drones_config, num_targets)
    
    # Run each policy
    engagement_tables = {}
    for policy_type, policy in policies.items():
        printer.policy_run_header(policy_type)
        policy_metrics_manager = MetricsManager(config.environment.mode)
        policy_episode_metrics: List[EpisodeMetrics] = []
        
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
            
            # This ensures that ENV noise/ordering is reproducible across runs
            episode_seed = config.seed + episode_num if config.seed is not None else None

            # Determine flush interval for continuous mode
            flush_interval = None
            if config.environment.mode == "continuous":
                flush_interval = config.environment.continuous.logging_interval_steps

            environment_logger.configure_continuous_flush(
                episode_num=episode_num,
                learning_state_provider=(
                    (
                        lambda: attach_target_classes_to_learning_state(
                            policy.get_learning_state(),
                            env,
                        )
                    )
                    if not is_deterministic
                    else None
                ),
                num_agents=getattr(policy, "num_agents", len(drones_config)),
                num_targets=getattr(policy, "num_targets", len(targets_config)),
                latent_dim=getattr(policy, "latent_dim", None),
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
            all_metrics.append(metrics)
            policy_episode_metrics.append(metrics)
            
            # Capture the latest post-episode state for CF policies
            episode_state = None
            if not is_deterministic:
                episode_state = attach_target_classes_to_learning_state(
                    policy.get_learning_state(),
                    env,
                )
                # Persist learning state for offline enrichment (e.g. t-SNE)
                environment_logger.save_learning_state(
                    episode_state=episode_state,
                    episode_num=episode_num,
                    num_agents=getattr(policy, "num_agents", len(drones_config)),
                    num_targets=getattr(policy, "num_targets", len(targets_config)),
                    latent_dim=getattr(policy, "latent_dim", None),
                )
            
            # Log metrics to episode log
            environment_logger.log_metrics(metrics)
            
            # Persist per-episode outputs via canonical logger entrypoint
            environment_logger.persist_episode_outputs(
                episode_num=episode_num,
                steps=metrics.steps,
            )
            
            # Per-run summary
            if config.environment.mode == "continuous":
                printer.continuous_run_progress(
                    steps=metrics.steps,
                    throughput=metrics.throughput,
                    coordination=metrics.coordination_str,
                    ammo_eff=metrics.ammo_eff,
                    collisions=metrics.total_collisions,
                )
            else:
                printer.episode_run_progress(
                    episode_num=episode_num,
                    steps=metrics.steps,
                    targets_neutralized=metrics.targets_neutralized,
                    total_net_damage=metrics.total_net_damage,
                    total_overkill=metrics.total_overkill,
                    total_reward=metrics.total_reward,
                )
            
            # Debug: Analyze agent clustering for CF policies
            # if False and not is_deterministic:
            #     drone_weapons = [d["weapon_type"] for d in drones_config]
            #     analyze_agent_clustering(policy, drone_weapons)
        
        # Save policy artifacts for episodic or continuous mode
        result = environment_logger.save_policy_episodes()

        if config.environment.mode != "continuous":
            printer.saved_episodes(result["files"])
        steps = result['steps']
        
        # Enrich all learning_state artifacts for matrix factorization with t-SNE
        # as a post-processing step, keeping the training loop itself fast.
        if (not is_deterministic and 
            policy_type == "matrix_factorization_cf" and
            config.collaborative_filtering and
            config.collaborative_filtering.enable_tsne_enrichment):
            learning_state_dir = environment_logger.get_learning_state_dir()
            tsne_mode = TSNE_MODE_PER_EPISODE #TSNE_MODE_PER_EPISODE_ALIGNED
            state_files = _iter_learning_state_files(learning_state_dir)

            if state_files:
                print("Starting t-SNE enrichment for all learning_state artifacts...")
                enrich_learning_state_dir(learning_state_dir, mode=tsne_mode)
                print("Finished t-SNE enrichment for all learning_state artifacts.")

        if 'final' in steps:
            # For continuous mode or learning policy final: show cumulative results
            printer.policy_final_summary(
                steps=metrics.steps,
                targets_neutralized=metrics.targets_neutralized,
                total_net_damage=metrics.total_net_damage,
                total_overkill=metrics.total_overkill,
                total_collisions=metrics.total_collisions,
            )
        elif 'first' in steps:
            printer.policy_first_step(steps["first"])
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

        policy_summaries[policy_type] = policy_metrics_manager.calc_total_episodes_metrics(
            policy_episode_metrics,
        )

    # show_policy_performance_summary(config, policy_summaries)

    show_policy_best_episode_performance_vs_random(config, policy_summaries)

    # show_engagement_tables(engagement_tables)

    printer.demo_complete()


if __name__ == "__main__":
    main()
