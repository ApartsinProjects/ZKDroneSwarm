"""
Tests for DroneEngageSingleTarget-v0 environment.
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tabula_drone.envs.drone_engage_single_target_v0 import (
    DroneEngageSingleTargetV0,
)
from tabula_drone.core.states import (
    DroneState,
    TargetState,
    WorldState,
    DEFAULT_CLASS_HP_MAPPING,
)


class TestStateDataclasses:
    """Test state dataclass instantiation and properties."""

    def test_drone_state_instantiation(self):
        """Test that DroneState can be instantiated with proper types."""
        drone = DroneState(
            id="drone_1",
            position=(100.0, 200.0),
            ammo=10,
            ammo_max=10,
            damage_per_shot=30.0
        )
        
        assert drone.id == "drone_1"
        assert drone.position == (100.0, 200.0)
        assert drone.ammo == 10
        assert drone.ammo_max == 10
        assert drone.damage_per_shot == 30.0

    def test_target_state_instantiation(self):
        """Test that TargetState can be instantiated with proper types."""
        target = TargetState(
            id="target_1",
            position=(500.0, 600.0),
            class_type="A",
            zone_id="zone_1",
            hp_initial=100.0,
            hp_current=100.0,
            is_active=True
        )
        
        assert target.id == "target_1"
        assert target.position == (500.0, 600.0)
        assert target.class_type == "A"
        assert target.zone_id == "zone_1"
        assert target.hp_initial == 100.0
        assert target.hp_current == 100.0
        assert target.is_active is True

    def test_world_state_instantiation(self):
        """Test that WorldState can be instantiated with proper types."""
        world = WorldState(
            world_size=(1000.0, 1000.0),
            time_step=0,
            max_steps=100,
            scenario_id="test_scenario",
            seed=42
        )
        
        assert world.world_size == (1000.0, 1000.0)
        assert world.time_step == 0
        assert world.max_steps == 100
        assert world.scenario_id == "test_scenario"
        assert world.seed == 42

    def test_world_state_optional_seed(self):
        """Test that WorldState can be instantiated without seed."""
        world = WorldState(
            world_size=(1000.0, 1000.0),
            time_step=0,
            max_steps=100,
            scenario_id="test_scenario"
        )
        
        assert world.seed is None

    def test_target_state_inactive(self):
        """Test target state with zero HP."""
        target = TargetState(
            id="target_1",
            position=(500.0, 600.0),
            class_type="A",
            zone_id="zone_1",
            hp_initial=100.0,
            hp_current=0.0,
            is_active=False
        )
        
        assert target.hp_current == 0.0
        assert target.is_active is False


class TestEnvironmentSkeleton:
    """Test environment class instantiation and spaces."""

    def test_environment_instantiation_defaults(self):
        """Test that environment can be instantiated with default parameters."""
        env = DroneEngageSingleTargetV0()
        
        assert isinstance(env, gym.Env)
        assert env.world_size == (1000.0, 1000.0)
        assert env.max_steps == 100
        assert env.scenario_id == "default"
        assert env.drone_position == (100.0, 100.0)
        assert env.drone_ammo_max == 10
        assert env.drone_damage_per_shot == 30.0
        assert env.target_position == (500.0, 500.0)
        assert env.target_class_type == "A"
        assert env.target_zone_id == "zone_1"
        assert env.target_hp_initial == 100.0

    def test_environment_instantiation_custom(self):
        """Test that environment can be instantiated with custom parameters."""
        env = DroneEngageSingleTargetV0(
            world_size=(2000.0, 2000.0),
            max_steps=200,
            drone_position=(200.0, 200.0),
            drone_ammo_max=20,
            drone_damage_per_shot=50.0,
            target_position=(800.0, 800.0),
            target_class_type="B",
            target_zone_id="zone_2",
            scenario_id="custom_test",
        )
        
        assert env.world_size == (2000.0, 2000.0)
        assert env.max_steps == 200
        assert env.drone_position == (200.0, 200.0)
        assert env.drone_ammo_max == 20
        assert env.drone_damage_per_shot == 50.0
        assert env.target_position == (800.0, 800.0)
        assert env.target_class_type == "B"
        assert env.target_zone_id == "zone_2"
        assert env.scenario_id == "custom_test"
        assert env.target_hp_initial == 150.0  # Class B maps to 150

    def test_observation_space_definition(self):
        """Test that observation space is correctly defined."""
        env = DroneEngageSingleTargetV0()
        
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (4,)
        assert env.observation_space.dtype == np.float32
        
        # Check bounds
        expected_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        expected_high = np.array([1.0, 1.0, np.inf, 1.0], dtype=np.float32)
        
        assert np.array_equal(env.observation_space.low, expected_low)
        assert np.array_equal(env.observation_space.high, expected_high)

    def test_action_space_definition(self):
        """Test that action space is correctly defined."""
        env = DroneEngageSingleTargetV0()
        
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 2

    def test_class_hp_mapping_default(self):
        """Test that default class-to-HP mapping is used."""
        env = DroneEngageSingleTargetV0()
        
        assert env.class_hp_mapping == DEFAULT_CLASS_HP_MAPPING
        assert env.class_hp_mapping["A"] == 100.0
        assert env.class_hp_mapping["B"] == 150.0
        assert env.class_hp_mapping["C"] == 200.0

    def test_class_hp_mapping_custom(self):
        """Test that custom class-to-HP mapping can be provided."""
        custom_mapping = {"A": 50.0, "B": 75.0, "C": 100.0}
        env = DroneEngageSingleTargetV0(
            target_class_type="C",
            class_hp_mapping=custom_mapping
        )
        
        assert env.class_hp_mapping == custom_mapping
        assert env.target_hp_initial == 100.0  # Class C with custom mapping

    def test_state_initialization_pending(self):
        """Test that state objects are None before reset."""
        env = DroneEngageSingleTargetV0()
        
        assert env.drone is None
        assert env.target is None
        assert env.world is None

    def test_observation_space_contains_samples(self):
        """Test that observation space can generate samples."""
        env = DroneEngageSingleTargetV0()
        
        # Sample should be valid
        sample = env.observation_space.sample()
        assert env.observation_space.contains(sample)


class TestResetLogic:
    """Test reset method and initial state."""

    def test_reset_returns_valid_tuple(self):
        """Test that reset returns (observation, info) tuple."""
        env = DroneEngageSingleTargetV0()
        result = env.reset()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        observation, info = result
        assert isinstance(observation, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_shape_and_type(self):
        """Test that observation has correct shape and dtype."""
        env = DroneEngageSingleTargetV0()
        observation, _ = env.reset()
        
        assert observation.shape == (4,)
        assert observation.dtype == np.float32

    def test_reset_observation_in_bounds(self):
        """Test that initial observation values are within expected bounds."""
        env = DroneEngageSingleTargetV0()
        observation, _ = env.reset()
        
        ammo_norm, hp_norm, distance, time_progress = observation
        
        # All normalized values should be in [0, 1]
        assert 0.0 <= ammo_norm <= 1.0
        assert 0.0 <= hp_norm <= 1.0
        assert 0.0 <= time_progress <= 1.0
        
        # Distance should be non-negative
        assert distance >= 0.0

    def test_reset_initial_values(self):
        """Test that initial observation has expected starting values."""
        env = DroneEngageSingleTargetV0()
        observation, _ = env.reset()
        
        ammo_norm, hp_norm, distance, time_progress = observation
        
        # At start: full ammo, full HP, time=0
        assert ammo_norm == 1.0
        assert hp_norm == 1.0
        assert time_progress == 0.0
        
        # Distance should match positions (default: (100,100) to (500,500))
        expected_distance = np.sqrt((100-500)**2 + (100-500)**2)
        assert np.isclose(distance, expected_distance)

    def test_reset_initializes_drone_state(self):
        """Test that reset properly initializes drone state."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        assert env.drone is not None
        assert env.drone.id == "drone_1"
        assert env.drone.position == (100.0, 100.0)
        assert env.drone.ammo == 10
        assert env.drone.ammo_max == 10
        assert env.drone.damage_per_shot == 30.0

    def test_reset_initializes_target_state(self):
        """Test that reset properly initializes target state."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        assert env.target is not None
        assert env.target.id == "target_1"
        assert env.target.position == (500.0, 500.0)
        assert env.target.class_type == "A"
        assert env.target.zone_id == "zone_1"
        assert env.target.hp_initial == 100.0
        assert env.target.hp_current == 100.0
        assert env.target.is_active is True

    def test_reset_initializes_world_state(self):
        """Test that reset properly initializes world state."""
        env = DroneEngageSingleTargetV0()
        observation, _ = env.reset(seed=42)
        
        assert env.world is not None
        assert env.world.world_size == (1000.0, 1000.0)
        assert env.world.time_step == 0
        assert env.world.max_steps == 100
        assert env.world.scenario_id == "default"
        assert env.world.seed == 42

    def test_reset_info_dict(self):
        """Test that reset returns correct info dictionary."""
        env = DroneEngageSingleTargetV0()
        _, info = env.reset()
        
        assert "step_index" in info
        assert "scenario_id" in info
        assert info["step_index"] == 0
        assert info["scenario_id"] == "default"

    def test_reset_with_seed(self):
        """Test that reset accepts and uses seed parameter."""
        env = DroneEngageSingleTargetV0()
        observation1, _ = env.reset(seed=42)
        observation2, _ = env.reset(seed=42)
        
        # Same seed should produce same initial state
        assert np.array_equal(observation1, observation2)

    def test_reset_multiple_times(self):
        """Test that environment can be reset multiple times."""
        env = DroneEngageSingleTargetV0()
        
        obs1, info1 = env.reset(seed=1)
        obs2, info2 = env.reset(seed=2)
        
        # Each reset should produce valid output
        assert obs1.shape == (4,)
        assert obs2.shape == (4,)
        assert info1["step_index"] == 0
        assert info2["step_index"] == 0

    def test_reset_custom_configuration(self):
        """Test reset with custom environment configuration."""
        env = DroneEngageSingleTargetV0(
            drone_position=(200.0, 300.0),
            target_position=(600.0, 700.0),
            drone_ammo_max=5,
            target_class_type="B",
        )
        observation, _ = env.reset()
        
        # Check state initialized with custom values
        assert env.drone.position == (200.0, 300.0)
        assert env.drone.ammo == 5
        assert env.target.position == (600.0, 700.0)
        assert env.target.hp_initial == 150.0  # Class B
        
        # Check observation reflects custom configuration
        ammo_norm, hp_norm, distance, time_progress = observation
        assert ammo_norm == 1.0  # Still full ammo
        assert hp_norm == 1.0  # Still full HP
        expected_distance = np.sqrt((200-600)**2 + (300-700)**2)
        assert np.isclose(distance, expected_distance)


class TestIdleActionDynamics:
    """Test Idle action (action=0) behavior."""

    def test_idle_step_returns_valid_tuple(self):
        """Test that step returns 5-element tuple."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        result = env.step(0)
        
        assert isinstance(result, tuple)
        assert len(result) == 5
        observation, reward, terminated, truncated, info = result
        assert isinstance(observation, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_idle_increments_time_step(self):
        """Test that Idle action increments time_step."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        initial_time = env.world.time_step
        env.step(0)
        
        assert env.world.time_step == initial_time + 1

    def test_idle_no_state_changes(self):
        """Test that Idle action doesn't change ammo or HP."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        initial_ammo = env.drone.ammo
        initial_hp = env.target.hp_current
        initial_active = env.target.is_active
        
        env.step(0)
        
        assert env.drone.ammo == initial_ammo
        assert env.target.hp_current == initial_hp
        assert env.target.is_active == initial_active

    def test_idle_updates_observation(self):
        """Test that Idle action updates time_progress in observation."""
        env = DroneEngageSingleTargetV0(max_steps=10)
        obs1, _ = env.reset()
        
        # Initial time_progress should be 0
        assert obs1[3] == 0.0
        
        obs2, _, _, _, _ = env.step(0)
        
        # After 1 step, time_progress should be 1/10 = 0.1
        assert obs2[3] == 0.1
        
        # Ammo and HP should remain at 1.0
        assert obs2[0] == 1.0  # ammo_norm
        assert obs2[1] == 1.0  # hp_norm

    def test_idle_reward_is_zero(self):
        """Test that Idle action returns zero reward."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        _, reward, _, _, _ = env.step(0)
        
        assert reward == 0.0

    def test_idle_not_terminated(self):
        """Test that Idle action doesn't terminate episode."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        _, _, terminated, _, _ = env.step(0)
        
        assert terminated is False

    def test_idle_truncation_at_max_steps(self):
        """Test that episode truncates when max_steps is reached."""
        env = DroneEngageSingleTargetV0(max_steps=5)
        env.reset()
        
        # Take 4 Idle steps (time_step will be 1, 2, 3, 4)
        for i in range(4):
            _, _, _, truncated, _ = env.step(0)
            assert truncated is False, f"Should not truncate at step {i+1}"
        
        # 5th step should trigger truncation (time_step = 5 >= max_steps)
        _, _, _, truncated, _ = env.step(0)
        assert truncated is True

    def test_idle_info_dict_updated(self):
        """Test that info dict reflects current step_index."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        _, _, _, _, info1 = env.step(0)
        assert info1["step_index"] == 1
        
        _, _, _, _, info2 = env.step(0)
        assert info2["step_index"] == 2

    def test_idle_multiple_steps(self):
        """Test multiple consecutive Idle actions."""
        env = DroneEngageSingleTargetV0(max_steps=10)
        env.reset()
        
        for step in range(5):
            obs, reward, terminated, truncated, info = env.step(0)
            
            assert env.world.time_step == step + 1
            assert reward == 0.0
            assert terminated is False
            assert truncated is False
            assert info["step_index"] == step + 1
            assert obs[3] == (step + 1) / 10  # time_progress

    def test_invalid_action_raises_error(self):
        """Test that invalid actions raise ValueError."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        with pytest.raises(ValueError, match="Invalid action.*2"):
            env.step(2)
        
        with pytest.raises(ValueError, match="Invalid action.*-1"):
            env.step(-1)


class TestFireActionDynamics:
    """Test Fire action (action=1) behavior."""

    def test_fire_decrements_ammo(self):
        """Test that Fire action decrements ammo."""
        env = DroneEngageSingleTargetV0(drone_ammo_max=10)
        env.reset()
        
        initial_ammo = env.drone.ammo
        env.step(1)  # Fire
        
        assert env.drone.ammo == initial_ammo - 1

    def test_fire_reduces_target_hp(self):
        """Test that Fire action reduces target HP."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        initial_hp = env.target.hp_current
        env.step(1)  # Fire
        
        assert env.target.hp_current == initial_hp - 30.0

    def test_fire_deactivates_target_when_hp_zero(self):
        """Test that Fire sets is_active=False when HP reaches 0."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        assert env.target.is_active is True
        env.step(1)  # Fire (100 damage to 100 HP)
        
        assert env.target.hp_current == 0.0
        assert env.target.is_active is False

    def test_fire_clamps_hp_to_zero(self):
        """Test that HP is clamped to 0, not negative (overkill damage)."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=150.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        env.step(1)  # Fire (150 damage to 100 HP)
        
        assert env.target.hp_current == 0.0  # Not -50
        assert env.target.is_active is False

    def test_fire_with_zero_ammo_no_effect(self):
        """Test that Fire with ammo=0 has no effect."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=1,
            drone_damage_per_shot=30.0
        )
        env.reset()
        
        # Fire once to deplete ammo
        env.step(1)
        assert env.drone.ammo == 0
        
        # Try to fire again with ammo=0
        initial_hp = env.target.hp_current
        env.step(1)  # Fire with ammo=0
        
        assert env.drone.ammo == 0  # Still 0
        assert env.target.hp_current == initial_hp  # HP unchanged

    def test_fire_on_inactive_target_no_effect(self):
        """Test that Fire on inactive target has no effect."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env.reset()
        
        # Neutralize target
        env.step(1)  # Fire
        assert env.target.is_active is False
        assert env.target.hp_current == 0.0
        
        initial_ammo = env.drone.ammo
        
        # Try to fire on inactive target
        env.step(1)  # Fire on inactive
        
        # Ammo should not be consumed
        assert env.drone.ammo == initial_ammo
        assert env.target.hp_current == 0.0

    def test_fire_updates_observation(self):
        """Test that Fire updates ammo_norm and hp_norm in observation."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        obs1, _ = env.reset()
        
        # Initial: ammo_norm=1.0, hp_norm=1.0
        assert obs1[0] == 1.0  # ammo_norm
        assert obs1[1] == 1.0  # hp_norm
        
        obs2, _, _, _, _ = env.step(1)  # Fire
        
        # After fire: ammo_norm=0.9 (9/10), hp_norm=0.7 (70/100)
        assert obs2[0] == 0.9  # ammo_norm
        assert obs2[1] == 0.7  # hp_norm

    def test_fire_multiple_shots_to_neutralize(self):
        """Test multiple Fire actions to neutralize target."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        # Fire 1: 100 -> 70 HP
        env.step(1)
        assert env.target.hp_current == 70.0
        assert env.target.is_active is True
        assert env.drone.ammo == 9
        
        # Fire 2: 70 -> 40 HP
        env.step(1)
        assert env.target.hp_current == 40.0
        assert env.target.is_active is True
        assert env.drone.ammo == 8
        
        # Fire 3: 40 -> 10 HP
        env.step(1)
        assert env.target.hp_current == 10.0
        assert env.target.is_active is True
        assert env.drone.ammo == 7
        
        # Fire 4: 10 -> 0 HP (neutralized with overkill)
        env.step(1)
        assert env.target.hp_current == 0.0
        assert env.target.is_active is False
        assert env.drone.ammo == 6

    def test_fire_increments_time_step(self):
        """Test that Fire action increments time_step."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        initial_time = env.world.time_step
        env.step(1)  # Fire
        
        assert env.world.time_step == initial_time + 1

    def test_fire_reward_without_neutralization(self):
        """Test that Fire returns zero reward when target not neutralized."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        # Fire doesn't neutralize (100 -> 70)
        _, reward, _, _, _ = env.step(1)
        
        assert reward == 0.0
        assert env.target.is_active is True

    def test_fire_terminates_on_target_neutralization(self):
        """Test that Fire sets terminated=True when target neutralized."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env.reset()
        
        # Neutralize target
        _, _, terminated, _, _ = env.step(1)
        
        assert terminated is True
        assert env.target.is_active is False

    def test_mixed_idle_and_fire_actions(self):
        """Test mixing Idle and Fire actions."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=5,
            drone_damage_per_shot=30.0,
            target_class_type="A"
        )
        env.reset()
        
        # Idle
        env.step(0)
        assert env.drone.ammo == 5
        assert env.target.hp_current == 100.0
        
        # Fire
        env.step(1)
        assert env.drone.ammo == 4
        assert env.target.hp_current == 70.0
        
        # Idle
        env.step(0)
        assert env.drone.ammo == 4  # No change
        assert env.target.hp_current == 70.0  # No change
        
        # Fire
        env.step(1)
        assert env.drone.ammo == 3
        assert env.target.hp_current == 40.0


class TestTerminationLogic:
    """Test termination conditions."""

    def test_terminated_on_target_neutralized(self):
        """Test that terminated=True when target is neutralized."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        _, _, terminated, truncated, _ = env.step(1)  # Fire to neutralize
        
        assert terminated is True
        assert truncated is False
        assert env.target.is_active is False

    def test_terminated_on_no_ammo_with_active_target(self):
        """Test that terminated=True when ammo runs out with active target."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=1,
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        # Fire once (ammo becomes 0, target still active at 70 HP)
        _, _, terminated, truncated, _ = env.step(1)
        
        assert terminated is True
        assert truncated is False
        assert env.drone.ammo == 0
        assert env.target.is_active is True

    def test_not_terminated_when_conditions_not_met(self):
        """Test that terminated=False when target active and ammo remains."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        # Fire once (target damaged but still active, ammo remains)
        _, _, terminated, _, _ = env.step(1)
        
        assert terminated is False
        assert env.drone.ammo > 0
        assert env.target.is_active is True

    def test_idle_not_terminated(self):
        """Test that Idle action doesn't trigger termination."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        _, _, terminated, _, _ = env.step(0)  # Idle
        
        assert terminated is False

    def test_simultaneous_termination_and_truncation_target_neutralized(self):
        """Test priority: terminated over truncated when target neutralized at max_steps."""
        env = DroneEngageSingleTargetV0(
            max_steps=1,
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env.reset()
        
        # At step 1 (max_steps), neutralize target
        _, _, terminated, truncated, _ = env.step(1)
        
        # Spec: prioritize terminated=True when target neutralized
        assert terminated is True
        assert truncated is False  # Should be False even though max_steps reached

    def test_simultaneous_termination_and_truncation_no_ammo(self):
        """Test both flags when no ammo at max_steps (no priority rule)."""
        env = DroneEngageSingleTargetV0(
            max_steps=1,
            drone_ammo_max=1,
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        # At step 1 (max_steps), fire but don't neutralize (ammo becomes 0)
        _, _, terminated, truncated, _ = env.step(1)
        
        # Both conditions are true (no ammo AND max_steps)
        # No priority rule for this case, both can be True
        assert terminated is True  # No ammo with active target
        assert truncated is True   # Max steps reached

    def test_termination_prevents_further_steps(self):
        """Test behavior after termination (environment can still step but should be reset)."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env.reset()
        
        # Neutralize target
        _, _, terminated1, _, _ = env.step(1)
        assert terminated1 is True
        
        # Environment can still step (Gymnasium doesn't prevent this)
        # but terminated should remain True
        _, _, terminated2, _, _ = env.step(0)  # Idle after termination
        assert terminated2 is True  # Still terminated

    def test_multiple_termination_paths(self):
        """Test that different paths to termination work correctly."""
        # Path 1: Neutralize target with plenty of ammo
        env1 = DroneEngageSingleTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env1.reset()
        _, _, term1, _, _ = env1.step(1)
        assert term1 is True
        assert env1.drone.ammo == 9  # Plenty left
        
        # Path 2: Run out of ammo without neutralizing
        env2 = DroneEngageSingleTargetV0(
            drone_ammo_max=2,
            drone_damage_per_shot=30.0,
            target_class_type="A"
        )
        env2.reset()
        env2.step(1)  # Fire 1: 100 -> 70
        _, _, term2, _, _ = env2.step(1)  # Fire 2: 70 -> 40, ammo=0
        assert term2 is True
        assert env2.target.is_active is True

    def test_idle_until_truncation_not_terminated(self):
        """Test that Idle until max_steps doesn't set terminated."""
        env = DroneEngageSingleTargetV0(max_steps=3)
        env.reset()
        
        # Idle for 3 steps
        _, _, term1, trunc1, _ = env.step(0)
        assert term1 is False and trunc1 is False
        
        _, _, term2, trunc2, _ = env.step(0)
        assert term2 is False and trunc2 is False
        
        _, _, term3, trunc3, _ = env.step(0)
        assert term3 is False  # Not terminated
        assert trunc3 is True  # Truncated at max_steps


class TestRewardLogic:
    """Test reward function."""

    def test_reward_on_target_neutralization(self):
        """Test that +1.0 reward is given when target neutralized."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        _, reward, _, _, _ = env.step(1)  # Fire to neutralize
        
        assert reward == 1.0
        assert env.target.is_active is False

    def test_reward_zero_without_neutralization(self):
        """Test that 0.0 reward when target not neutralized."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        _, reward, _, _, _ = env.step(1)  # Fire (100 -> 70)
        
        assert reward == 0.0
        assert env.target.is_active is True

    def test_reward_zero_on_idle(self):
        """Test that Idle action gives 0.0 reward."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        _, reward, _, _, _ = env.step(0)  # Idle
        
        assert reward == 0.0

    def test_reward_only_on_neutralization_step(self):
        """Test that reward is given only on the step where neutralization occurs."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        # Fire 1: 100 -> 70 (no reward)
        _, reward1, _, _, _ = env.step(1)
        assert reward1 == 0.0
        
        # Fire 2: 70 -> 40 (no reward)
        _, reward2, _, _, _ = env.step(1)
        assert reward2 == 0.0
        
        # Fire 3: 40 -> 10 (no reward)
        _, reward3, _, _, _ = env.step(1)
        assert reward3 == 0.0
        
        # Fire 4: 10 -> 0 (REWARD!)
        _, reward4, _, _, _ = env.step(1)
        assert reward4 == 1.0

    def test_reward_with_overkill_damage(self):
        """Test that reward is given even with overkill damage."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=150.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        # Fire with overkill (150 damage to 100 HP)
        _, reward, _, _, _ = env.step(1)
        
        assert reward == 1.0
        assert env.target.hp_current == 0.0

    def test_reward_zero_when_ammo_depleted(self):
        """Test that no reward when ammo depletes without neutralizing."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=1,
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP
        )
        env.reset()
        
        # Fire once (ammo becomes 0, target still alive at 70 HP)
        _, reward, terminated, _, _ = env.step(1)
        
        assert reward == 0.0
        assert terminated is True  # Episode ends (no ammo)
        assert env.target.is_active is True

    def test_reward_zero_after_neutralization(self):
        """Test that subsequent steps after neutralization give 0.0 reward."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env.reset()
        
        # Neutralize target
        _, reward1, _, _, _ = env.step(1)
        assert reward1 == 1.0
        
        # Try to step again (shouldn't happen in practice but test behavior)
        # Target was already inactive, so no new neutralization
        _, reward2, _, _, _ = env.step(0)  # Idle
        assert reward2 == 0.0  # No reward because target didn't just become neutralized

    def test_reward_zero_on_truncation(self):
        """Test that truncation without neutralization gives 0.0 reward."""
        env = DroneEngageSingleTargetV0(
            max_steps=2,
            drone_damage_per_shot=30.0,
            target_class_type="A"
        )
        env.reset()
        
        # Idle until truncation
        env.step(0)
        _, reward, terminated, truncated, _ = env.step(0)
        
        assert reward == 0.0
        assert terminated is False
        assert truncated is True

    def test_cumulative_reward_in_episode(self):
        """Test total reward for a successful neutralization episode."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=30.0,
            target_class_type="A"  # 100 HP, needs 4 shots
        )
        env.reset()
        
        total_reward = 0.0
        for i in range(4):
            _, reward, terminated, _, _ = env.step(1)  # Fire
            total_reward += reward
            if terminated:
                break
        
        # Only the last step should give reward
        assert total_reward == 1.0


class TestInfoDictionary:
    """Test info dictionary contents."""

    def test_info_basic_fields_always_present(self):
        """Test that basic fields are always in info dict."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        _, _, _, _, info = env.step(0)  # Idle
        
        assert "step_index" in info
        assert "scenario_id" in info
        assert info["step_index"] == 1
        assert info["scenario_id"] == "default"

    def test_info_debug_fields_always_present(self):
        """Test that optional debug fields are always included."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        _, _, _, _, info = env.step(0)  # Idle
        
        assert "ammo" in info
        assert "hp_current" in info
        assert "class_type" in info
        assert "zone_id" in info

    def test_info_done_reason_on_target_neutralized(self):
        """Test that done_reason is set when target neutralized."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env.reset()
        
        _, _, terminated, _, info = env.step(1)  # Fire to neutralize
        
        assert terminated is True
        assert "done_reason" in info
        assert info["done_reason"] == "target_neutralized"

    def test_info_done_reason_on_no_ammo(self):
        """Test that done_reason is set when ammo depleted."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=1,
            drone_damage_per_shot=30.0,
            target_class_type="A"
        )
        env.reset()
        
        _, _, terminated, _, info = env.step(1)  # Fire (ammo becomes 0)
        
        assert terminated is True
        assert "done_reason" in info
        assert info["done_reason"] == "no_ammo"

    def test_info_done_reason_on_max_steps(self):
        """Test that done_reason is set when max_steps reached."""
        env = DroneEngageSingleTargetV0(max_steps=2)
        env.reset()
        
        env.step(0)  # Idle step 1
        _, _, _, truncated, info = env.step(0)  # Idle step 2
        
        assert truncated is True
        assert "done_reason" in info
        assert info["done_reason"] == "max_steps"

    def test_info_no_done_reason_during_episode(self):
        """Test that done_reason is not present when episode ongoing."""
        env = DroneEngageSingleTargetV0()
        env.reset()
        
        _, _, terminated, truncated, info = env.step(0)  # Idle
        
        assert terminated is False
        assert truncated is False
        assert "done_reason" not in info

    def test_info_debug_fields_accuracy(self):
        """Test that debug fields reflect actual state."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=30.0,
            target_class_type="B",
            target_zone_id="zone_2"
        )
        env.reset()
        
        # After Fire: ammo=9, hp=120 (150-30)
        _, _, _, _, info = env.step(1)
        
        assert info["ammo"] == 9
        assert info["hp_current"] == 120.0
        assert info["class_type"] == "B"
        assert info["zone_id"] == "zone_2"

    def test_info_debug_fields_after_neutralization(self):
        """Test debug fields when target neutralized."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=5,
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env.reset()
        
        _, _, _, _, info = env.step(1)  # Fire to neutralize
        
        assert info["ammo"] == 4
        assert info["hp_current"] == 0.0
        assert info["done_reason"] == "target_neutralized"

    def test_info_complete_structure_on_done(self):
        """Test that all expected fields present when episode ends."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env.reset()
        
        _, _, _, _, info = env.step(1)  # Fire to neutralize
        
        # Required fields
        assert "step_index" in info
        assert "scenario_id" in info
        assert "done_reason" in info
        
        # Debug fields
        assert "ammo" in info
        assert "hp_current" in info
        assert "class_type" in info
        assert "zone_id" in info

    def test_info_done_reason_priority(self):
        """Test done_reason priority when multiple conditions."""
        # Target neutralized at max_steps (should be target_neutralized)
        env = DroneEngageSingleTargetV0(
            max_steps=1,
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        env.reset()
        
        _, _, terminated, truncated, info = env.step(1)
        
        # Prioritizes target_neutralized over max_steps
        assert info["done_reason"] == "target_neutralized"
        assert terminated is True
        assert truncated is False  # Priority rule applies


class TestIntegration:
    """Integration tests for full episode trajectories."""

    def test_full_episode_successful_neutralization(self):
        """Test a complete successful episode from start to finish."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=30.0,
            target_class_type="A",  # 100 HP
            max_steps=20
        )
        
        obs, info = env.reset(seed=42)
        
        # Validate initial state
        assert obs[0] == 1.0  # Full ammo
        assert obs[1] == 1.0  # Full HP
        assert obs[3] == 0.0  # Time = 0
        
        # Execute episode
        total_reward = 0.0
        step_count = 0
        
        for i in range(4):
            obs, reward, terminated, truncated, info = env.step(1)  # Fire
            total_reward += reward
            step_count += 1
            
            if terminated:
                # Should terminate on 4th shot
                assert i == 3
                assert info["done_reason"] == "target_neutralized"
                assert reward == 1.0
                assert env.target.hp_current == 0.0
                assert env.target.is_active is False
                break
        
        assert step_count == 4
        assert total_reward == 1.0
        assert terminated is True
        assert truncated is False

    def test_full_episode_ammo_depletion_failure(self):
        """Test a complete failed episode (ammo runs out)."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=3,
            drone_damage_per_shot=30.0,
            target_class_type="A",  # 100 HP, needs 4 shots
            max_steps=20
        )
        
        obs, info = env.reset(seed=42)
        
        total_reward = 0.0
        for i in range(3):
            obs, reward, terminated, truncated, info = env.step(1)  # Fire
            total_reward += reward
            
            if i < 2:
                assert terminated is False
            else:
                # Last shot depletes ammo but doesn't neutralize
                assert terminated is True
                assert info["done_reason"] == "no_ammo"
                assert env.drone.ammo == 0
                assert env.target.is_active is True
                assert env.target.hp_current == 10.0  # 100 - 3*30
        
        assert total_reward == 0.0  # No reward for failure

    def test_full_episode_truncation(self):
        """Test a complete episode ending by truncation."""
        env = DroneEngageSingleTargetV0(
            max_steps=5,
            drone_damage_per_shot=30.0,
            target_class_type="A"
        )
        
        obs, info = env.reset(seed=42)
        
        total_reward = 0.0
        for i in range(5):
            obs, reward, terminated, truncated, info = env.step(0)  # Idle
            total_reward += reward
            
            if i < 4:
                assert terminated is False
                assert truncated is False
            else:
                assert terminated is False
                assert truncated is True
                assert info["done_reason"] == "max_steps"
        
        assert total_reward == 0.0

    def test_determinism_same_seed(self):
        """Test that same seed produces identical trajectories."""
        seed = 42
        
        # Episode 1
        env1 = DroneEngageSingleTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=30.0,
            target_class_type="A"
        )
        obs1, _ = env1.reset(seed=seed)
        trajectory1 = [obs1.copy()]
        
        for i in range(4):
            obs, _, _, _, _ = env1.step(1)  # Fire
            trajectory1.append(obs.copy())
        
        # Episode 2
        env2 = DroneEngageSingleTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=30.0,
            target_class_type="A"
        )
        obs2, _ = env2.reset(seed=seed)
        trajectory2 = [obs2.copy()]
        
        for i in range(4):
            obs, _, _, _, _ = env2.step(1)  # Fire
            trajectory2.append(obs.copy())
        
        # Trajectories should be identical
        assert len(trajectory1) == len(trajectory2)
        for t1, t2 in zip(trajectory1, trajectory2):
            assert np.array_equal(t1, t2)

    def test_mixed_action_sequence(self):
        """Test a complex sequence of mixed Idle and Fire actions."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=5,
            drone_damage_per_shot=30.0,
            target_class_type="A",
            max_steps=20
        )
        
        obs, info = env.reset()
        
        # Sequence: Idle, Fire, Idle, Fire, Fire, Idle, Fire (neutralize)
        actions = [0, 1, 0, 1, 1, 0, 1]
        expected_ammo = [5, 4, 4, 3, 2, 2, 1]
        expected_hp = [100, 70, 70, 40, 10, 10, 0]
        
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert env.drone.ammo == expected_ammo[i]
            assert env.target.hp_current == expected_hp[i]
            
            if i < len(actions) - 1:
                assert reward == 0.0
                assert terminated is False
            else:
                assert reward == 1.0
                assert terminated is True
                assert info["done_reason"] == "target_neutralized"

    def test_observation_bounds_throughout_episode(self):
        """Test that observations stay within valid bounds during episode."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=5,
            drone_damage_per_shot=30.0,
            target_class_type="A",
            max_steps=10
        )
        
        obs, _ = env.reset()
        
        # Check initial observation
        assert env.observation_space.contains(obs)
        
        # Check observations during episode
        for i in range(6):
            obs, _, terminated, _, _ = env.step(1)  # Fire
            
            # All components should be in valid ranges
            assert 0.0 <= obs[0] <= 1.0  # ammo_norm
            assert 0.0 <= obs[1] <= 1.0  # hp_norm
            assert obs[2] >= 0.0  # distance
            assert 0.0 <= obs[3] <= 1.0  # time_progress
            
            # Should be valid according to space
            assert env.observation_space.contains(obs)
            
            if terminated:
                break

    def test_multiple_episodes_without_reset(self):
        """Test that environment behaves correctly across multiple episodes."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"
        )
        
        # Episode 1
        obs1, info1 = env.reset(seed=1)
        _, _, term1, _, info1_final = env.step(1)  # Fire to neutralize
        assert term1 is True
        assert info1_final["done_reason"] == "target_neutralized"
        
        # Episode 2 (should reset properly)
        obs2, info2 = env.reset(seed=2)
        
        # State should be reset
        assert env.drone.ammo == env.drone.ammo_max
        assert env.target.hp_current == env.target.hp_initial
        assert env.target.is_active is True
        assert env.world.time_step == 0
        
        # Can complete another episode
        _, _, term2, _, info2_final = env.step(1)  # Fire to neutralize
        assert term2 is True
        assert info2_final["done_reason"] == "target_neutralized"

    def test_gymnasium_api_compliance(self):
        """Test that environment follows Gymnasium API correctly."""
        env = DroneEngageSingleTargetV0()
        
        # Check that env is a Gymnasium environment
        assert isinstance(env, gym.Env)
        
        # Check spaces
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'action_space')
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(env.action_space, spaces.Discrete)
        
        # Check reset signature
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        
        # Check step signature
        result = env.step(0)
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_edge_case_exact_neutralization(self):
        """Test edge case where damage exactly equals remaining HP."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=100.0,
            target_class_type="A"  # Exactly 100 HP
        )
        
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # Fire
        
        assert env.target.hp_current == 0.0  # Exactly zero
        assert env.target.is_active is False
        assert terminated is True
        assert reward == 1.0
        assert info["done_reason"] == "target_neutralized"

    def test_edge_case_overkill_by_large_margin(self):
        """Test edge case with massive overkill damage."""
        env = DroneEngageSingleTargetV0(
            drone_damage_per_shot=1000.0,
            target_class_type="A"  # 100 HP
        )
        
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # Fire
        
        # HP should clamp to 0, not go negative
        assert env.target.hp_current == 0.0
        assert env.target.is_active is False
        assert terminated is True
        assert reward == 1.0

    def test_state_consistency_throughout_episode(self):
        """Test that state remains consistent throughout episode."""
        env = DroneEngageSingleTargetV0(
            drone_ammo_max=5,
            drone_damage_per_shot=30.0,
            target_class_type="A"
        )
        
        env.reset()
        
        for i in range(4):
            # Check invariants before step
            assert env.drone.ammo >= 0
            assert env.drone.ammo <= env.drone.ammo_max
            assert env.target.hp_current >= 0.0
            assert env.target.hp_current <= env.target.hp_initial
            assert (env.target.is_active and env.target.hp_current > 0) or \
                   (not env.target.is_active and env.target.hp_current == 0)
            
            obs, reward, terminated, truncated, info = env.step(1)  # Fire
            
            # Check invariants after step
            assert env.drone.ammo >= 0
            assert env.target.hp_current >= 0.0
            
            if terminated:
                break
