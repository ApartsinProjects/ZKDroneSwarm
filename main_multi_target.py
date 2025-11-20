"""
Main demonstration script for DroneEngageMultiTarget-v0 environment.

Executes a single episode with a simple policy and detailed console logging.
"""

from tabula_drone.envs.drone_engage_multi_target_v0 import DroneEngageMultiTargetV0


def main():
    """Run a single episode with a simple targeting policy and display results."""
    
    # Create environment with multiple targets
    env = DroneEngageMultiTargetV0(
        world_size=(1000.0, 1000.0),
        max_steps=100,
        drone_position=(500.0, 500.0),
        drone_ammo_max=15,
        drone_damage_per_shot=35.0,
        targets_config=[
            {'position': (200.0, 200.0), 'class_type': 'A', 'zone_id': 'zone_north'},
            {'position': (800.0, 200.0), 'class_type': 'B', 'zone_id': 'zone_east'},
            {'position': (200.0, 800.0), 'class_type': 'C', 'zone_id': 'zone_south'},
            {'position': (800.0, 800.0), 'class_type': 'A', 'zone_id': 'zone_west'},
        ],
        scenario_id='multi_target_demo',
    )
    
    print("=" * 80)
    print("DroneEngageMultiTarget-v0 Episode Demonstration")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  World Size: {env.world_size}")
    print(f"  Drone Position: {env.drone_position}")
    print(f"  Ammo: {env.drone_ammo_max}")
    print(f"  Damage per Shot: {env.drone_damage_per_shot}")
    print(f"  Max Steps: {env.max_steps}")
    print(f"  Targets: {len(env.targets_config)}")
    print()
    
    # Display target information
    print("Target Configuration:")
    for i, target_cfg in enumerate(env.targets_config):
        hp = env.class_hp_mapping[target_cfg['class_type']]
        print(f"  Target {i}: Class {target_cfg['class_type']} (HP: {hp}), "
              f"Position: {target_cfg['position']}, Zone: {target_cfg['zone_id']}")
    print()
    
    print(f"Action Space: {env.action_space} (0=Idle, 1-{len(env.targets_config)}=Fire at target)")
    print(f"Observation Space: {env.observation_space}")
    print()
    
    # Initialize episode
    observation, info = env.reset(seed=42)
    terminated = False
    truncated = False
    
    print(f"{'Step':<6} {'Action':<15} {'Reward':<8} {'Ammo':<6} {'Target Status':<40} {'Done':<10}")
    print("-" * 100)
    
    total_reward = 0.0
    episode_step = 0
    
    # Episode loop with simple policy: fire at first active target
    while not (terminated or truncated):
        episode_step += 1
        
        # Simple policy: fire at first active target, or idle if all neutralized
        action = 0  # Default: Idle
        for i, target in enumerate(env.targets):
            if target.is_active:
                action = i + 1  # Fire at this target
                break
        
        # Execute step
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Format action label
        if action == 0:
            action_label = "Idle"
        else:
            target_idx = action - 1
            action_label = f"Fire → T{target_idx}"
        
        # Format target status
        target_status = ""
        for i, (hp, active) in enumerate(zip(info['target_hps'], info['target_active'])):
            status = "ACTIVE" if active else "DOWN"
            target_status += f"T{i}:{hp:>5.0f}/{env.class_hp_mapping[info['target_classes'][i]]:<3.0f}({status}) "
        
        # Format done status
        if terminated or truncated:
            done_status = f"✓ {info.get('done_reason', 'done')}"
        else:
            done_status = ""
        
        # Print step information
        print(f"{episode_step:<6} {action_label:<15} {reward:<8.1f} {info['ammo']:<6} {target_status:<40} {done_status:<10}")
    
    # Episode summary
    print()
    print("=" * 80)
    print("Episode Summary")
    print("=" * 80)
    print(f"Done Reason: {info.get('done_reason', 'unknown')}")
    print(f"Total Steps: {episode_step}")
    print(f"Total Reward: {total_reward:.1f}")
    print(f"Final Ammo: {info['ammo']}/{env.drone_ammo_max}")
    print()
    print(f"Final Target Status:")
    for i, (hp, active, class_type, zone) in enumerate(zip(
        info['target_hps'], 
        info['target_active'], 
        info['target_classes'],
        info['target_zones']
    )):
        status = "✓ Neutralized" if not active else f"  Active ({hp}/{env.class_hp_mapping[class_type]} HP)"
        print(f"  Target {i} (Class {class_type}, {zone}): {status}")
    
    neutralized_count = sum(1 for active in info['target_active'] if not active)
    print()
    print(f"Targets Neutralized: {neutralized_count}/{len(env.targets_config)}")
    
    if neutralized_count == len(env.targets_config):
        print("Result: ✓ MISSION SUCCESS - All targets neutralized!")
    else:
        print(f"Result: ✓ PARTIAL SUCCESS - {neutralized_count} of {len(env.targets_config)} targets neutralized")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
