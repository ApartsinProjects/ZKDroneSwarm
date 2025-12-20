"""
Demo script for ZK-MRTA environment with Random policy.

Demonstrates:
- Multi-agent PettingZoo environment setup
- Random policy baseline
- Episode execution
- Metrics collection and logging
"""

from typing import Dict, Any, List, Optional, Union

from tabula_drone.config import load_config
from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA
from tabula_drone.logging import EpisodeLogger
from tabula_drone.policies.random_policy import RandomPolicy
from tabula_drone.policies.oracle_policy import OracleTimeToKillPolicy
from tabula_drone.scenarios import ScenarioBuilder

CONFIG_PATH = "config/scenario.json"


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


PolicyType = Union[RandomPolicy, OracleTimeToKillPolicy]


def run_episode(
    env: DroneEngageZKMRTA,
    policy: PolicyType,
    episode_num: int,
    verbose: bool = True,
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
        logger.start_episode(env, info, seed)
    
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
    
    # Episode loop
    while not done:
        step_count += 1
        
        # Policy selects actions for all agents
        if isinstance(policy, OracleTimeToKillPolicy):
            actions = policy.select_actions(obs, env.num_targets, info["target_attributes"])
        else:
            actions = policy.select_actions(obs, env.num_targets)
        
        # Environment step
        obs, rewards, terminations, truncations, info = env.step(actions)
        
        # Check termination
        terminated = terminations[env.agents[0]]
        truncated = truncations[env.agents[0]]
        
        if logger:
            logger.log_step(step_count, actions, rewards, terminated, truncated, info)
        
        # Update total rewards
        for agent_id in env.agents:
            total_rewards[agent_id] += rewards[agent_id]
        
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
    
    # Finalize logger
    if logger:
        logger.end_episode(total_rewards, info.get("done_reason"))
        logger.save()
    
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
    }


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
    
    # Create environment
    env = DroneEngageZKMRTA(
        world_size=config.world.size,
        max_steps=config.environment.max_steps,
        drones_config=drones_config,
        targets_config=targets_config,
        scenario_id=config.environment.scenario_id,
        class_attribute_mapping=config.mappings.class_attribute_mapping,
        weapon_damage_profile_mapping=config.mappings.weapon_damage_profile_mapping,
        policy_type=config.policy.type,
    )
    
    # Create policy based on config type
    if config.policy.type == "oracle":
        # Oracle needs a weapon damage profile - use first weapon type from config
        # In multi-weapon scenarios, Oracle uses a representative profile
        first_weapon_type = list(config.mappings.weapon_damage_profile_mapping.keys())[0]
        weapon_profile = config.mappings.weapon_damage_profile_mapping[first_weapon_type]
        policy: PolicyType = OracleTimeToKillPolicy(
            weapon_damage_profile=weapon_profile,
            seed=config.seed,
            allow_noop=config.policy.allow_noop,
        )
    else:
        policy = RandomPolicy(seed=config.seed, allow_noop=config.policy.allow_noop)
    
    # Run episodes
    num_episodes = config.execution.num_episodes
    
    print("\n" + "="*60)
    print("ZK-MRTA ENVIRONMENT DEMO")
    print("="*60)
    print(f"Config File: {CONFIG_PATH}")
    print(f"Environment: {env.metadata['name']}")
    print(f"Scenario ID: {env.scenario_id}")
    print(f"World Size: {env.world_size}")
    print(f"Max Steps: {env.max_steps}")
    print(f"Random Seed: {config.seed}")
    print(f"Policy: {policy.__class__.__name__}")
    print(f"Episodes: {num_episodes}")
    print(f"Weapon Damage Profiles: {config.mappings.weapon_damage_profile_mapping}")
    print(f"Target Class Attributes: {config.mappings.class_attribute_mapping}")
    print("="*60)
    all_metrics = []

    logger = EpisodeLogger(output_dir=config.logging.output_dir)

    for episode_num in range(1, num_episodes + 1):
        metrics = run_episode(
            env=env,
            policy=policy,
            episode_num=episode_num,
            verbose=config.execution.verbose,
            logger=logger,
            seed=config.seed
        )
        all_metrics.append(metrics)
    
    # Aggregate statistics across episodes
    print("\n" + "="*60)
    print("AGGREGATE STATISTICS")
    print("="*60)
    print(f"Total Episodes: {num_episodes}")
    
    avg_steps = sum(m["steps"] for m in all_metrics) / num_episodes
    avg_targets = sum(m["targets_neutralized"] for m in all_metrics) / num_episodes
    avg_ammo = sum(m["total_ammo_used"] for m in all_metrics) / num_episodes
    avg_overkill = sum(m["total_overkill"] for m in all_metrics) / num_episodes
    
    print(f"Average Steps:              {avg_steps:.1f}")
    print(f"Average Targets Neutralized: {avg_targets:.1f}")
    print(f"Average Ammo Used:          {avg_ammo:.1f}")
    print(f"Average Overkill Damage:    {avg_overkill:.1f}")
    
    # Per-agent statistics
    print(f"\nPer-Agent Average Rewards:")
    for agent_id in env.agents:
        avg_reward = sum(m["agent_rewards"][agent_id] for m in all_metrics) / num_episodes
        print(f"  {agent_id}: {avg_reward:.2f}")
    
    print("="*60)
    print("\nDemo complete! ✓")


if __name__ == "__main__":
    main()
