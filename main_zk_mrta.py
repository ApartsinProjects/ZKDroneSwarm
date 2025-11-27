"""
Demo script for ZK-MRTA environment with Random policy.

Demonstrates:
- Multi-agent PettingZoo environment setup
- Random policy baseline
- Episode execution
- Metrics collection and logging
"""

from typing import Dict, Any, List

from tabula_drone.envs.drone_engage_zk_mrta_v0 import DroneEngageZKMRTA
from tabula_drone.policies.random_policy import RandomPolicy
from tabula_drone.scenarios import assign_weapons_to_drones


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
    print(f"\nAgent Rewards:")
    for agent_id, reward in sorted(total_rewards.items()):
        ammo = info['ammo_used'][agent_id]
        print(f"  {agent_id}: {reward:.1f} (ammo: {ammo})")
    print("=" * 60 + "\n")


def run_episode(
    env: DroneEngageZKMRTA,
    policy: RandomPolicy,
    episode_num: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single episode with the given policy.
    
    Args:
        env: ZK-MRTA environment
        policy: Policy for action selection
        episode_num: Episode number for logging
        verbose: If True, print step-by-step details
    
    Returns:
        Episode metrics dictionary
    """
    # Reset environment
    obs, info = env.reset()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_num} START")
        print(f"{'='*60}")
        print(f"Drones: {env.num_drones}, Targets: {env.num_targets}")
        print(f"Target classes: {info['target_classes']}")
        print(f"Target zones: {info['target_zones']}")
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
        actions = policy.select_actions(obs, env.num_targets)
        
        # Environment step
        obs, rewards, terminations, truncations, info = env.step(actions)
        
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
        done = terminations[env.agents[0]] or truncations[env.agents[0]]
    
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
    
    # Environment configuration
    # Define drone positions
    base_drones_config = [
        {"position": (100.0, 100.0)},
        {"position": (200.0, 200.0)},
        # {"position": (300.0, 300.0)},
    ]
    
    # Assign weapons using weighted distribution
    weapon_distribution = {
        "light": 0.2,
        "medium": 0.5,
        "heavy": 0.3,
    }
    drones_config = assign_weapons_to_drones(
        base_drones_config,
        distribution=weapon_distribution,
        seed=42  # For reproducibility
    )
    
    targets_config = [
        {"position": (500.0, 500.0), "class_type": "A", "zone_id": "zone_1"},
        {"position": (700.0, 700.0), "class_type": "B", "zone_id": "zone_2"},
        {"position": (900.0, 900.0), "class_type": "C", "zone_id": "zone_3"},
    ]
    
    # Create environment
    env = DroneEngageZKMRTA(
        world_size=(1000.0, 1000.0),
        max_steps=50,
        drones_config=drones_config,
        targets_config=targets_config,
        scenario_id="random_policy_demo",
    )
    
    print("\n" + "="*60)
    print("ZK-MRTA ENVIRONMENT DEMO")
    print("="*60)
    print(f"Environment: {env.metadata['name']}")
    print(f"Num Drones: {env.num_drones}")
    print(f"Num Targets: {env.num_targets}")
    print(f"Weapon Types: {[d['weapon_type'] for d in drones_config]}")
    print(f"Max Steps: {env.max_steps}")
    print("="*60)
    
    # Create policy
    policy = RandomPolicy(seed=42, allow_noop=False)
    
    # Run episodes
    num_episodes = 1
    all_metrics = []
    
    for episode_num in range(1, num_episodes + 1):
        metrics = run_episode(
            env=env,
            policy=policy,
            episode_num=episode_num,
            verbose=True,
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
