"""
Main demonstration script for DroneEngageSingleTarget-v0 environment.

Executes a single episode with random action selection and console logging.
"""

from tabula_drone.envs.drone_engage_single_target_v0 import DroneEngageSingleTargetV0


def main():
    """Run a single episode with random actions and display results."""
    
    # Create environment with hardcoded parameters
    env = DroneEngageSingleTargetV0(
        drone_ammo_max=5,
        drone_damage_per_shot=35.0,
        target_class_type="A",  # 100 HP
        max_steps=50,
    )
    
    print("=" * 60)
    print("DroneEngageSingleTarget-v0 Episode Demonstration")
    print("=" * 60)
    print(f"Configuration: Ammo={env.drone_ammo_max}, Damage={env.drone_damage_per_shot}, "
          f"Target Class={env.target_class_type}, Max Steps={env.max_steps}")
    print()
    
    # Initialize episode
    observation, info = env.reset()
    terminated = False
    truncated = False
    
    print(f"{'Step':<6} {'Action':<12} {'Ammo':<8} {'HP':<8} {'Distance':<10} {'Time':<6} {'Reward':<8} {'Status'}")
    print("-" * 80)
    
    # Episode loop
    while not (terminated or truncated):
        # Sample random action
        action = env.action_space.sample()
        
        # Execute step
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Extract observation components
        ammo_norm = observation[0]
        hp_norm = observation[1]
        distance = observation[2]
        time_progress = observation[3]
        
        # Format action label
        action_label = "Idle" if action == 0 else "Fire"
        
        # Print step information
        step_num = info['step_index']
        status = ""
        if terminated:
            status = "TERMINATED"
        elif truncated:
            status = "TRUNCATED"
        
        print(f"{step_num:<6} {action_label:<12} {ammo_norm:<8.2f} {hp_norm:<8.2f} "
              f"{distance:<10.1f} {time_progress:<6.3f} {reward:<8.1f} {status}")
    
    # Episode summary
    print()
    print("=" * 60)
    print("Episode Summary")
    print("=" * 60)
    print(f"Done Reason: {info.get('done_reason', 'unknown')}")
    print(f"Total Steps: {info['step_index']}")
    print(f"Final Ammo: {info['ammo']}/{env.drone_ammo_max}")
    print(f"Final Target HP: {info['hp_current']:.1f}/{env.target_hp_initial}")
    print(f"Target Neutralized: {'Yes' if info['hp_current'] == 0.0 else 'No'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
