"""
Demo script for ZK-MRTA environment with Random policy.

Demonstrates:
- Multi-agent PettingZoo environment setup
- Random policy baseline
- Episode execution
- Metrics collection and logging
"""

from typing import Dict, Any, List, Optional, Union

import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

from tabula_drone.config import load_config
from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA, REWARD_MODE
from tabula_drone.logging import EpisodeLogger, RunManager
from tabula_drone.logging.engagement_summary import (
    build_target_x_drone_table,
    extract_drone_engagement_counts,
    load_analysis,
)
from tabula_drone.policies.random_policy import RandomPolicy
from tabula_drone.policies.min_ttk_oracle import OracleTimeToKillPolicy
from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
from tabula_drone.scenarios import ScenarioBuilder
from tabula_drone.policies.ucb_cf_policy import UCBCFPolicy
from tabula_drone.policies.selfish_ep_greedy_cf_policy import SelfishEpGreedyCFPolicy
from tabula_drone.policies.coordinated_ep_greedy_cf_policy import CoordinatedEpGreedyCFPolicy
from tabula_drone.policies.multi_agent_policy import MultiAgentPolicy
from tabula_drone.policies.base import IPolicy

CONFIG_PATH = "config/scenario.json"

# "type": ["max_damage_oracle", "min_ttk_oracle", "ep_greedy_cf", "ucb_cf", "random"],
def print_episode_summary(
    episode_num: int,
    step_count: int,
    total_rewards: Dict[str, float],
    info: Dict[str, Any],
    targets_neutralized: int,
    total_ammo_used: int,
) -> None:
    """Print summary statistics for a completed episode."""
    print("\n" + "=" * 60)
    print(f"EPISODE {episode_num} SUMMARY")
    print("=" * 60)
    print(f"Done Reason:          {info.get('done_reason', 'N/A')}")
    print(f"Steps:                {step_count}")
    print(f"Targets Neutralized:  {targets_neutralized}")
    print(f"Total Ammo Used:      {total_ammo_used}")
    print(f"\nAgent Performance:")
    for agent_id, reward in sorted(total_rewards.items()):
        ammo = info['ammo_used'][agent_id]
        # Extract weapon type for this agent (drone_0 -> index 0, etc.)
        agent_idx = int(agent_id.split('_')[1])
        weapon_type = info['weapon_types'][agent_idx]
        print(f"  {agent_id}: {reward:.1f} (weapon: {weapon_type}, ammo: {ammo})")
    print("=" * 60 + "\n")


def print_learning_path(
    policy: Union["UCBCFPolicy", Dict[str, "SelfishEpGreedyCFPolicy"]],
    drones_config: List[Dict[str, Any]],
    targets_config: List[Dict[str, Any]],
) -> None:
    """Print current latent vectors for CF policy learning visualization."""
    print("    --- Learning Path ---")
    
    # Handle decentralized policy (dict of policy instances)
    if isinstance(policy, dict):
        first_policy = next(iter(policy.values()))
        latent_dim = first_policy.latent_dim
        # Collect agent vectors from each agent's private state
        agent_lv = [policy[f"drone_{i}"].agent_lv for i in range(len(policy))]
        # Use first agent's target estimates (they may differ between agents)
        target_lv = first_policy.target_lv
    else:
        latent_dim = policy.latent_dim
        agent_lv = policy.agent_lv
        target_lv = policy.target_lv
    
    # Agent latent vectors
    agent_headers = ["Agent", "Weapon"] + [f"d{i}" for i in range(latent_dim)]
    agent_rows = []
    for i, drone_cfg in enumerate(drones_config):
        row = [f"A{i}", drone_cfg["weapon_type"][:4]]
        row.extend([f"{v:.3f}" for v in agent_lv[i]])
        agent_rows.append(row)
    agent_rows_sorted = sorted(agent_rows, key=lambda r: r[1])
    print(tabulate(agent_rows_sorted, headers=agent_headers, tablefmt="simple"))
    
    print()  # Separator
    
    # Target latent vectors
    target_headers = ["Target", "Class"] + [f"d{i}" for i in range(latent_dim)]
    target_rows = []
    for i, target_cfg in enumerate(targets_config):
        row = [f"T{i}", target_cfg["class_type"][:4]]
        row.extend([f"{v:.3f}" for v in target_lv[i]])
        target_rows.append(row)
    target_rows_sorted = sorted(target_rows, key=lambda r: r[1])
    print(tabulate(target_rows_sorted, headers=target_headers, tablefmt="simple"))
    print()


def print_target_class_profile(
    class_attribute_mapping: Dict[str, Dict[str, float]],
) -> None:
    """Print target class HP attributes profile."""
    print("\nTarget Class Profile (HP):")
    attributes = list(next(iter(class_attribute_mapping.values())).keys())
    attr_short = [a[:6] for a in attributes]
    class_headers = ["Class"] + attr_short + ["Dominant Attr"]
    class_rows = []
    for cls, attrs in sorted(class_attribute_mapping.items()):
        dominant_attr = max(attrs.items(), key=lambda x: x[1])[0]
        row = [cls] + [int(attrs[a]) for a in attributes] + [dominant_attr[:6]]
        class_rows.append(row)
    print(tabulate(class_rows, headers=class_headers, tablefmt="simple"))
    print()


def print_weapon_damage_profile(
    weapon_damage_profile_mapping: Dict[str, Dict[str, float]],
    class_attribute_mapping: Dict[str, Dict[str, float]],
) -> None:
    """Print weapon damage profile per attribute."""
    print("\nWeapon Damage Profile:")
    attributes = list(next(iter(class_attribute_mapping.values())).keys())
    attr_short = [a[:6] for a in attributes]
    weapon_headers = ["Weapon"] + attr_short
    weapon_rows = []
    for weapon, profile in sorted(weapon_damage_profile_mapping.items()):
        row = [weapon] + [int(profile[a]) for a in attributes]
        weapon_rows.append(row)
    print(tabulate(weapon_rows, headers=weapon_headers, tablefmt="simple"))
    print()


def print_drone_setup(
    drones_config: List[Dict[str, Any]],
) -> None:
    """Print drone setup showing ID and weapon assignment."""
    print("\nDrone Setup:")
    drone_headers = ["Drone", "Weapon"]
    drone_rows = [[f"D{i}", cfg["weapon_type"]] for i, cfg in enumerate(drones_config)]
    print(tabulate(drone_rows, headers=drone_headers, tablefmt="simple"))
    print()


def print_target_setup(
    targets_config: List[Dict[str, Any]],
    class_attribute_mapping: Dict[str, Dict[str, float]],
) -> None:
    """Print target setup showing ID, class, and dominant attribute."""
    print("\nTarget Setup:")
    target_headers = ["Target", "Class", "Dominant Attr"]
    target_rows = []
    for i, target_cfg in enumerate(targets_config):
        class_type = target_cfg["class_type"]
        attributes = class_attribute_mapping[class_type]
        dominant_attr = max(attributes.items(), key=lambda x: x[1])[0]
        row = [f"T{i}", class_type, dominant_attr[:6]]
        target_rows.append(row)
    print(tabulate(target_rows, headers=target_headers, tablefmt="simple"))
    print()


def print_optimal_engagement_prediction(
    drones_config: List[Dict[str, Any]],
    targets_config: List[Dict[str, Any]],
    class_attribute_mapping: Dict[str, Dict[str, float]],
    weapon_damage_profile_mapping: Dict[str, Dict[str, float]],
) -> None:
    """Print optimal engagement prediction based on weapon-target attribute matching."""
    print("\nOptimal Engagement Prediction (Greedy):")
    
    # Calculate damage efficiency for each drone-target pair
    assignments = []
    for drone_idx, drone_cfg in enumerate(drones_config):
        weapon_type = drone_cfg["weapon_type"]
        weapon_damage = weapon_damage_profile_mapping[weapon_type]
        
        for target_idx, target_cfg in enumerate(targets_config):
            class_type = target_cfg["class_type"]
            target_attrs = class_attribute_mapping[class_type]
            dominant_attr = max(target_attrs.items(), key=lambda x: x[1])[0]
            
            # Calculate damage to dominant attribute
            damage_to_dominant = weapon_damage[dominant_attr]
            total_target_hp = sum(target_attrs.values())
            
            # Efficiency score: damage to dominant attribute
            efficiency = damage_to_dominant
            
            assignments.append({
                'drone_id': f"D{drone_idx}",
                'target_id': f"T{target_idx}",
                'weapon': weapon_type,
                'target_class': class_type,
                'dominant_attr': dominant_attr[:6],
                'damage': damage_to_dominant,
                'efficiency': efficiency
            })
    
    # Greedy assignment: best match first
    assigned_drones = set()
    assigned_targets = set()
    optimal_assignments = []
    
    # Sort by efficiency (descending)
    assignments.sort(key=lambda x: x['efficiency'], reverse=True)
    
    for assignment in assignments:
        if assignment['drone_id'] not in assigned_drones and assignment['target_id'] not in assigned_targets:
            optimal_assignments.append(assignment)
            assigned_drones.add(assignment['drone_id'])
            assigned_targets.add(assignment['target_id'])
            
            if len(optimal_assignments) == min(len(drones_config), len(targets_config)):
                break
    
    # Display prediction table
    headers = ["Drone", "→", "Target", "Target Class", "Weapon", "Target Attr", "Damage"]
    rows = []
    for assignment in sorted(optimal_assignments, key=lambda x: x['drone_id']):
        rows.append([
            assignment['drone_id'],
            "→",
            assignment['target_id'],
            assignment['target_class'],
            assignment['weapon'][:6],
            assignment['dominant_attr'],
            int(assignment['damage'])
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()


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
    logger: Optional[EpisodeLogger] = None,
    seed: Optional[int] = None,
    total_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run a single episode with the given policy.
    
    Args:
        env: ZK-MRTA environment
        policy: Policy for action selection
        episode_num: Episode number for logging
        verbose: If True, print step-by-step details
        logger: Optional EpisodeLogger for capturing episode data
        seed: Random seed used for this episode (for logger)
    
    Returns:
        Episode metrics dictionary
    """
    # Reset environment
    obs, info = env.reset()
    
    if logger:
        logger.start_episode(env, info, seed, episode_num, total_episodes)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_num} START")
        print(f"{'='*60}")
        print(f"Drones: {env.num_drones}, Targets: {env.num_targets}")
        print(f"Target classes: {info['target_classes']}")
        print(f"Drone weapons: {info['weapon_types']}")
        print(f"Initial HPs: {info['target_hps']}")
        print()
    
    # Initialize tracking
    total_rewards = {agent_id: 0.0 for agent_id in env.agents}
    step_count = 0
    done = False
    overkill_events: List[Dict[int, float]] = []
    total_effective_damage = 0.0
    
    # Episode loop
    while not done:
        step_count += 1
        
        # Policy selects actions for all agents (uniform interface)
        actions = policy.select_actions(obs, info)
        policy.update(obs)
        
        # Environment step
        obs, rewards, terminations, truncations, info = env.step(actions)
        
        # Check termination
        terminated = terminations[env.agents[0]]
        truncated = truncations[env.agents[0]]
        
        if logger:
            logger.log_step(step_count, actions, rewards, terminated, truncated, info)
        
        # Update total rewards and track effective damage
        for agent_id in env.agents:
            total_rewards[agent_id] += rewards[agent_id]
            # Positive rewards represent actual_damage / 10.0
            if rewards[agent_id] > 0:
                total_effective_damage += rewards[agent_id] * 10.0
        
        # Track overkill
        if "overkill" in info:
            overkill_events.append(info["overkill"])
        
        # Verbose logging
        if verbose:
            print(f"Step {step_count}:")
            print(f"  Actions: {actions}")
            print(f"  Target HPs: {info['target_hps']}")
            print(f"  Target Active: {info['target_active']}")
            print(f"  Step Rewards: {rewards}")
            
            if "overkill" in info:
                print(f"  Overkill: {info['overkill']}")
        
        # Check termination
        done = terminated or truncated
    
    # Finalize logger (save is handled by caller for best-episode tracking)
    if logger:
        logger.end_episode(total_rewards, info.get("done_reason"))
    
    # Compute final metrics
    targets_neutralized = sum(1 for active in info['target_active'] if not active)
    total_ammo_used = sum(info['ammo_used'].values())
    total_overkill = sum(
        sum(overkill.values()) for overkill in overkill_events
    )
    
    # Print summary
    if verbose:
        print_episode_summary(
            episode_num,
            step_count,
            total_rewards,
            info,
            targets_neutralized,
            total_ammo_used,
        )
    
    # Compute potential damage from actual weapon profiles and ammo used per agent
    total_potential_damage = 0.0
    for agent_id, ammo in info['ammo_used'].items():
        agent_idx = int(agent_id.split('_')[1])
        weapon_type = info['weapon_types'][agent_idx]
        damage_per_shot = sum(env.weapon_damage_profile_mapping[weapon_type].values())
        total_potential_damage += ammo * damage_per_shot
    
    # Return metrics
    return {
        "episode": episode_num,
        "steps": step_count,
        "targets_neutralized": targets_neutralized,
        "total_ammo_used": total_ammo_used,
        "total_overkill": total_overkill,
        "done_reason": info.get("done_reason"),
        "agent_rewards": total_rewards.copy(),
        "overkill_events": len(overkill_events),
        "total_effective_damage": total_effective_damage,
        "total_potential_damage": total_potential_damage,
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
    
    print("\n--- Agent Latent Vector Analysis ---")
    print("Agent Types:")
    for i, weapon in enumerate(drone_weapon_map):
        print(f"  Agent {i}: {weapon}")
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            w1 = drone_weapon_map[i]
            w2 = drone_weapon_map[j]
            similarity = sim_matrix[i, j]
            
            # Highlight if drones with the SAME weapon are becoming similar
            match_status = "[MATCH]" if w1 == w2 else "       "
            print(f"Agent {i}({w1[:4]}) vs Agent {j}({w2[:4]}): {similarity:.4f} {match_status}")


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
    num_episodes = config.execution.num_episodes
    all_metrics = []
    
    print("\n" + "="*60)
    print("ZK-MRTA ENVIRONMENT DEMO")
    print("="*60)
    print(f"Config File: {CONFIG_PATH}")
    print(f"Mappings File: {config.mappings_file}")
    print(f"World Size: {config.world.size}")
    print(f"Random Seed: {config.seed}")
    print(f"Policy Types: {config.policy.type}")
    print(f"Episodes per Policy: {num_episodes}")
    print_target_class_profile(
        config.mappings.class_attribute_mapping,
    )
    print_weapon_damage_profile(
        config.mappings.weapon_damage_profile_mapping,
        config.mappings.class_attribute_mapping,
    )
    print_drone_setup(
        drones_config,
    )
    print_target_setup(
        targets_config,
        config.mappings.class_attribute_mapping,
    )
    print_optimal_engagement_prediction(
        drones_config,
        targets_config,
        config.mappings.class_attribute_mapping,
        config.mappings.weapon_damage_profile_mapping,
    )
    print("="*60)
    
    # Create RunManager for structured logging
    run_manager = RunManager(
        output_dir=config.logging.output_dir,
        scenario_id=config.environment.scenario_id
    )
    
    # Determine if any policy requires CF noise settings
    has_cf_policy = any("cf" in policy_type for policy_type in config.policy.type)
    if has_cf_policy:
        # Use CF noise settings from config
        cf_config = getattr(config, 'collaborative_filtering', None)
        reward_noise = cf_config.reward_noise if cf_config else 0.1
        observation_noise = cf_config.observation_noise if cf_config else 0.05
    else:
        # No CF policies - use zero noise
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
        policy_type="random",  # Default value, not used for single env
        reward_noise=reward_noise,
        observation_noise=observation_noise,
    )
    
    # Create all policies upfront
    policies = create_all_policies(config, drones_config, num_targets)
    
    # Run each policy
    for policy_type, policy in policies.items():
        print(f"\n>>> Running policy: {policy_type}")
        
        # Get policy metadata from policy attributes
        is_deterministic = policy.is_deterministic
        effective_episodes = 1 if is_deterministic else num_episodes
        logger = EpisodeLogger(output_dir=config.logging.output_dir, policy_type=policy_type)
        
        # Start policy run in RunManager
        run_manager.start_policy(policy_type, is_deterministic=is_deterministic)
        
        for episode_num in range(1, effective_episodes + 1):
            # Soft reset CF policy for new episode (preserves agent latent vectors)
            if not is_deterministic and episode_num > 1:
                policy.soft_reset()
            
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
            
            metrics = run_episode(
                env=env,
                policy=policy,
                episode_num=episode_num,
                verbose=config.execution.verbose,
                logger=logger,
                seed=config.seed,
                total_episodes=num_episodes,
            )
            metrics["policy_type"] = policy_type
            all_metrics.append(metrics)
            
            # Capture post-episode state for CF policies
            if not is_deterministic:
                post_episode_state = policy.get_learning_state()
            
            # Compute alignment score for CF policies (used by both trackers)
            alignment_score = None
            if not is_deterministic:
                alignment_score = compute_alignment_score(
                    pre_episode_lv[0], pre_episode_lv[1], drones_config, targets_config
                )
            
            # Save learning state using RunManager (every episode)
            if not is_deterministic:
                run_manager.save_learning_state(
                    pre_state=pre_episode_state,
                    post_state=post_episode_state,
                    episode_num=episode_num,
                    num_agents=policy.num_agents,
                    num_targets=policy.num_targets,
                    latent_dim=policy.latent_dim,
                )
            
            # Get episode data
            episode_data = logger.to_dict()
            
            # Get analysis data for engagement tracking and save per-episode
            analysis_data = logger.get_analysis_data()
            run_manager.save_analysis(analysis_data, episode_num)
            
            # Record episode in RunManager for selection
            run_manager.record_episode(episode_data, metrics["steps"])
            
            # Per-episode summary
            total_reward = sum(metrics["agent_rewards"].values())
            print(f"  Episode {episode_num}: Steps={metrics['steps']}, "
                  f"Targets={metrics['targets_neutralized']}, "
                  f"Overkill={metrics['total_overkill']:.0f}, "
                  f"Reward={total_reward:.0f}")
            
            # Print learning path for CF policies
            if not is_deterministic and config.execution.verbose:
                print_learning_path(policy, drones_config, targets_config)
            
            # Debug: Analyze agent clustering for CF policies
            if False and not is_deterministic:
                drone_weapons = [d["weapon_type"] for d in drones_config]
                analyze_agent_clustering(policy, drone_weapons)
        
        # Finalize policy run - saves selected episodes (first/best/mid or only)
        result = run_manager.finalize_policy()
        print(f"  Saved episodes: {result['files']}")
        steps = result['steps']
        if 'best' in steps:
            print(f"  Steps: first={steps['first']}, best={steps['best']}, mid={steps['mid']}")
        else:
            print(f"  Steps: first={steps['first']}")

        best_episode_num = result.get("best_episode_num")
        if best_episode_num is not None:
            analysis_path = os.path.join(
                run_manager.get_analysis_dir(),
                f"analysis_ep{best_episode_num:02d}.json",
            )
            if os.path.exists(analysis_path):
                analysis_data = load_analysis(analysis_path)
                drone_engagement_counts = extract_drone_engagement_counts(analysis_data)
                engagement_headers, engagement_rows = build_target_x_drone_table(
                    drone_engagement_counts
                )
                print("\nActual Engagement Summary (Best Episode):")
                print(
                    tabulate(
                        engagement_rows,
                        headers=engagement_headers,
                        tablefmt="simple",
                    )
                )
                print()
            else:
                print("\nActual Engagement Summary (Best Episode):")
                print(f"Analysis file not found: {analysis_path}")
                print()

    if config.execution.verbose:
        # Aggregate statistics across all episodes (all policies)
        total_episodes = len(all_metrics)
        print("\n" + "="*60)
        print("AGGREGATE STATISTICS")
        print("="*60)
        print(f"Total Episodes: {total_episodes} ({num_episodes} per policy × {len(config.policy.type)} policies)")

        avg_steps = sum(m["steps"] for m in all_metrics) / total_episodes
        avg_targets = sum(m["targets_neutralized"] for m in all_metrics) / total_episodes
        avg_ammo = sum(m["total_ammo_used"] for m in all_metrics) / total_episodes
        avg_overkill = sum(m["total_overkill"] for m in all_metrics) / total_episodes

        print(f"Average Steps:              {avg_steps:.1f}")
        print(f"Average Targets Neutralized: {avg_targets:.1f}")
        print(f"Average Ammo Used:          {avg_ammo:.1f}")
        print(f"Average Overkill Damage:    {avg_overkill:.1f}")
    
        # Per-policy statistics
        for policy_type in config.policy.type:
            policy_metrics = [m for m in all_metrics if m["policy_type"] == policy_type]
            if policy_metrics:
                print(f"\n  Policy '{policy_type}':")
                print(f"    Avg Steps: {sum(m['steps'] for m in policy_metrics) / len(policy_metrics):.1f}")
                print(f"    Avg Targets: {sum(m['targets_neutralized'] for m in policy_metrics) / len(policy_metrics):.1f}")
    
        # Per-agent statistics (use last env for agent list)
        print(f"\nPer-Agent Average Rewards:")
        for agent_id in env.agents:
            avg_reward = sum(m["agent_rewards"][agent_id] for m in all_metrics) / total_episodes
            print(f"  {agent_id}: {avg_reward:.2f}")

        print("="*60)
    
    # Policy Performance Summary Table
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
    
    # Sort by Avg Steps ascending
    table_data.sort(key=lambda row: row[1])
    # Format Avg Steps for display after sorting
    for row in table_data:
        row[1] = f"{row[1]:.1f}"
    
    print("\n" + "="*60)
    print("POLICY PERFORMANCE SUMMARY")
    print("="*60)
    headers = ["Policy", "Avg Steps", "Avg Targets", "Avg Ammo", "Avg Overkill", "Avg Reward", "Success %", "Ammo Eff", "Dmg Eff"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print("="*60)
    
    # Policy Best Episode Performance (comparison to random baseline)
    policy_best_metrics = {}
    for policy_type in config.policy.type:
        policy_metrics = [m for m in all_metrics if m["policy_type"] == policy_type]
        if policy_metrics:
            best_ep = min(policy_metrics, key=lambda m: m["steps"])
            best_ammo = best_ep["total_ammo_used"]
            best_targets = best_ep["targets_neutralized"]
            best_eff_dmg = best_ep["total_effective_damage"]
            best_pot_dmg = best_ep["total_potential_damage"]
            policy_best_metrics[policy_type] = {
                "steps": best_ep["steps"],
                "ammo_eff": best_targets / best_ammo if best_ammo > 0 else 0.0,
                "dmg_eff": best_eff_dmg / best_pot_dmg if best_pot_dmg > 0 else 0.0,
                "shots_per_target": best_ammo / best_targets if best_targets > 0 else 0.0,
            }
    
    if "random" in policy_best_metrics and len(policy_best_metrics) > 1:
        print("\n" + "="*60)
        print("POLICY BEST EPISODE PERFORMANCE (vs Random Baseline)")
        print("="*60)
        print(f"Reward Mode: {REWARD_MODE}")
        print(f"Mappings File: {config.mappings_file}")
        print("Ammo Eff = targets / ammo (higher = fewer wasted shots)")
        print("Dmg Eff  = effective_dmg / potential_dmg (higher = less overkill)")
        rand = policy_best_metrics["random"]
        
        def fmt_pct(val, base, fmt, higher_better=True):
            if base == 0:
                return fmt.format(val)
            pct = ((val - base) / base) * 100
            if not higher_better:
                pct = -pct
            sign = "+" if pct > 0 else ""
            return f"{fmt.format(val)} ({sign}{pct:.0f}%)"
        
        cmp_data = []
        for pt in config.policy.type:
            if pt in policy_best_metrics:
                m = policy_best_metrics[pt]
                if pt == "random":
                    cmp_data.append([pt, f"{m['steps']} (baseline)", f"{m['shots_per_target']:.1f} (baseline)", f"{m['ammo_eff']:.3f} (baseline)", f"{m['dmg_eff']:.1%} (baseline)"])
                else:
                    cmp_data.append([pt, fmt_pct(m["steps"], rand["steps"], "{}", False), fmt_pct(m["shots_per_target"], rand["shots_per_target"], "{:.1f}", False), fmt_pct(m["ammo_eff"], rand["ammo_eff"], "{:.3f}", True), fmt_pct(m["dmg_eff"], rand["dmg_eff"], "{:.1%}", True)])
        
        cmp_data.sort(key=lambda r: int(r[1].split()[0]))
        print(tabulate(cmp_data, headers=["Policy", "Best Steps", "Shots/Target", "Ammo Eff", "Dmg Eff"], tablefmt="grid"))
        print("="*60)
    
    print("\nDemo complete! ✓")


if __name__ == "__main__":
    main()
