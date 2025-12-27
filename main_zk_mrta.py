"""
Demo script for ZK-MRTA environment with Random policy.

Demonstrates:
- Multi-agent PettingZoo environment setup
- Random policy baseline
- Episode execution
- Metrics collection and logging
"""

from typing import Dict, Any, List, Optional, Union
import copy

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

from tabula_drone.config import load_config
from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA
from tabula_drone.logging import EpisodeLogger
from tabula_drone.policies.random_policy import RandomPolicy
from tabula_drone.policies.min_ttk_oracle import OracleTimeToKillPolicy
from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle
from tabula_drone.scenarios import ScenarioBuilder
from tabula_drone.policies.ep_greedy_cf_policy import EpGreedyCFPolicy
from tabula_drone.policies.ucb_cf_policy import UCBCFPolicy

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
    policy: Union["EpGreedyCFPolicy", "UCBCFPolicy"],
    drones_config: List[Dict[str, Any]],
    targets_config: List[Dict[str, Any]],
) -> None:
    """Print current latent vectors for CF policy learning visualization."""
    print("    --- Learning Path ---")
    
    # Agent latent vectors
    agent_headers = ["Agent", "Weapon"] + [f"d{i}" for i in range(policy.latent_dim)]
    agent_rows = []
    for i, drone_cfg in enumerate(drones_config):
        row = [f"A{i}", drone_cfg["weapon_type"][:4]]
        row.extend([f"{v:.3f}" for v in policy.agent_lv[i]])
        agent_rows.append(row)
    print(tabulate(agent_rows, headers=agent_headers, tablefmt="simple"))
    
    print()  # Separator
    
    # Target latent vectors
    target_headers = ["Target", "Class"] + [f"d{i}" for i in range(policy.latent_dim)]
    target_rows = []
    for i, target_cfg in enumerate(targets_config):
        row = [f"T{i}", target_cfg["class_type"][:4]]
        row.extend([f"{v:.3f}" for v in policy.target_lv[i]])
        target_rows.append(row)
    print(tabulate(target_rows, headers=target_headers, tablefmt="simple"))
    print()


def print_scenario_config(
    drones_config: List[Dict[str, Any]],
    class_attribute_mapping: Dict[str, Dict[str, float]],
    weapon_damage_profile_mapping: Dict[str, Dict[str, float]],
) -> None:
    """Print scenario configuration tables for agents, target classes, and weapon profiles."""
    # Table 1: Agent Configuration
    print("\nAgent Configuration:")
    agent_headers = ["Agent", "Weapon"]
    agent_rows = [[f"A{i}", cfg["weapon_type"]] for i, cfg in enumerate(drones_config)]
    print(tabulate(agent_rows, headers=agent_headers, tablefmt="simple"))
    
    # Table 2: Target Class Attributes (HP)
    print("\nTarget Class Attributes (HP):")
    attributes = list(next(iter(class_attribute_mapping.values())).keys())
    attr_short = [a[:6] for a in attributes]  # Abbreviate column names
    class_headers = ["Class"] + attr_short
    class_rows = []
    for cls, attrs in sorted(class_attribute_mapping.items()):
        row = [cls] + [int(attrs[a]) for a in attributes]
        class_rows.append(row)
    print(tabulate(class_rows, headers=class_headers, tablefmt="simple"))
    
    # Table 3: Weapon Damage Profiles
    print("\nWeapon Damage Profiles:")
    weapon_headers = ["Weapon"] + attr_short
    weapon_rows = []
    for weapon, profile in sorted(weapon_damage_profile_mapping.items()):
        row = [weapon] + [int(profile[a]) for a in attributes]
        weapon_rows.append(row)
    print(tabulate(weapon_rows, headers=weapon_headers, tablefmt="simple"))
    print()


PolicyType = Union[RandomPolicy, OracleTimeToKillPolicy, OptimalAssignmentOracle, EpGreedyCFPolicy, UCBCFPolicy]


def create_policy(
    policy_type: str,
    config: Any,
    drones_config: List[Dict[str, Any]],
    num_targets: Optional[int] = None,
) -> PolicyType:
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
    elif policy_type == "ep_greedy_cf":
        if num_targets is None:
            raise ValueError("num_targets is required for ep_greedy_cf policy")
        return EpGreedyCFPolicy(
            num_agents=len(drones_config),
            num_targets=num_targets,
            seed=config.seed,
        )
    elif policy_type == "ucb_cf":
        if num_targets is None:
            raise ValueError("num_targets is required for ucb_cf policy")
        ucb_c = config.collaborative_filtering.ucb_c if config.collaborative_filtering else 2.0
        return UCBCFPolicy(
            num_agents=len(drones_config),
            num_targets=num_targets,
            seed=config.seed,
            ucb_c=ucb_c,
        )
    else:
        return RandomPolicy(seed=config.seed, allow_noop=config.policy.allow_noop)


def run_episode(
    env: DroneEngageZKMRTA,
    policy: PolicyType,
    episode_num: int,
    verbose: bool = False,
    logger: Optional[EpisodeLogger] = None,
    seed: Optional[int] = None,
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
        logger.start_episode(env, info, seed, episode_num)
    
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
        
        # Policy selects actions for all agents
        if isinstance(policy, (OracleTimeToKillPolicy, OptimalAssignmentOracle)):
            actions = policy.select_actions(obs, env.num_targets, info["target_attributes"])
        elif isinstance(policy, (EpGreedyCFPolicy, UCBCFPolicy)):
            actions = policy.select_actions(obs)
            # CF policy learns from observations
            for agent_id, agent_obs in obs.items():
                policy.update_from_observation(agent_obs, agent_id)
        else:
            actions = policy.select_actions(obs, env.num_targets)
        
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
    print(f"World Size: {config.world.size}")
    print(f"Random Seed: {config.seed}")
    print(f"Policy Types: {config.policy.type}")
    print(f"Episodes per Policy: {num_episodes}")
    print_scenario_config(
        drones_config,
        config.mappings.class_attribute_mapping,
        config.mappings.weapon_damage_profile_mapping,
    )
    print("="*60)
    
    # Run each policy type
    for policy_type in config.policy.type:
        print(f"\n>>> Running policy: {policy_type}")
        
        # Determine episode count and logging strategy based on policy type
        # Deterministic policies: run 1 episode (results are reproducible)
        # CF policies: run all episodes but only log the last one
        is_deterministic = policy_type in ("min_ttk_oracle", "max_damage_oracle", "random")
        is_cf = policy_type in ("ep_greedy_cf", "ucb_cf")
        effective_episodes = 1 if is_deterministic else num_episodes
        
        # Create environment with appropriate observation mode per policy
        
        env = DroneEngageZKMRTA(
            world_size=config.world.size,
            max_steps=config.environment.max_steps,
            drones_config=drones_config,
            targets_config=targets_config,
            scenario_id=config.environment.scenario_id,
            class_attribute_mapping=config.mappings.class_attribute_mapping,
            weapon_damage_profile_mapping=config.mappings.weapon_damage_profile_mapping,
            policy_type=policy_type,
            observation_mode="collaborative" if is_cf else "minimal",
            reward_noise=reward_noise if is_cf else 0.0,
            observation_noise=observation_noise if is_cf else 0.0,
        )
        
        # Create policy (pass num_targets for CF policy)
        policy = create_policy(policy_type, config, drones_config, num_targets=env.num_targets)
        logger = EpisodeLogger(output_dir=config.logging.output_dir, policy_type=policy_type)
        
        # Best-episode tracking per policy
        best_episode_data = None
        best_step_count = float('inf')
        
        for episode_num in range(1, effective_episodes + 1):
            # Soft reset CF policy for new episode (preserves agent latent vectors)
            if is_cf and episode_num > 1:
                policy.soft_reset()
            
            metrics = run_episode(
                env=env,
                policy=policy,
                episode_num=episode_num,
                verbose=config.execution.verbose,
                logger=logger,
                seed=config.seed
            )
            metrics["policy_type"] = policy_type
            all_metrics.append(metrics)
            
            # Track best episode (minimum steps)
            if metrics["steps"] < best_step_count:
                best_step_count = metrics["steps"]
                best_episode_data = copy.deepcopy(logger._episode_data)
            
            # Per-episode summary
            total_reward = sum(metrics["agent_rewards"].values())
            print(f"  Episode {episode_num}: Steps={metrics['steps']}, "
                  f"Targets={metrics['targets_neutralized']}, "
                  f"Overkill={metrics['total_overkill']:.0f}, "
                  f"Reward={total_reward:.0f}")
            
            # Print learning path for CF policies
            if is_cf and config.execution.verbose:
                print_learning_path(policy, drones_config, targets_config)
            
            # Debug: Analyze agent clustering for CF policies
            if False and is_cf:
                drone_weapons = [d["weapon_type"] for d in drones_config]
                analyze_agent_clustering(policy, drone_weapons)
        
        # Save only the best episode for this policy
        if best_episode_data is not None:
            logger._episode_data = best_episode_data
            saved_path = logger.save(is_best=True)
            best_ep_num = best_episode_data.get("episode_num", "?")
            print(f"  Saved best episode (ep{best_ep_num}, {best_step_count} steps): {saved_path}")
        
        # Print final Learning Path for CF policies (after all episodes)
        if is_cf:
            print(f"  --- Final Learning Path ({policy_type}) ---")
            print_learning_path(policy, drones_config, targets_config)

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
    
    print("\nDemo complete! ✓")


if __name__ == "__main__":
    main()
