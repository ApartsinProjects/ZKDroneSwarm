"""
Tests for DroneEngageMultiTarget-v0 environment.
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from tabula_drone.envs.drone_engage_multi_target_v0 import (
    DroneEngageMultiTargetV0,
)
from tabula_drone.core.states import (
    DroneState,
    TargetState,
    WorldState,
    DEFAULT_CLASS_HP_MAPPING,
)


class TestConstructorValidation:
    """Test constructor parameter validation."""

    def test_valid_configuration(self):
        """Test that environment can be instantiated with valid configuration."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (300, 400), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        assert isinstance(env, gym.Env)
        assert len(env.targets_config) == 2

    def test_empty_targets_config_raises_error(self):
        """Test that empty targets_config raises ValueError."""
        with pytest.raises(ValueError, match="at least one target"):
            DroneEngageMultiTargetV0(targets_config=[])

    def test_none_targets_config_raises_error(self):
        """Test that None targets_config raises ValueError."""
        with pytest.raises(ValueError, match="at least one target"):
            DroneEngageMultiTargetV0(targets_config=None)

    def test_missing_position_key_raises_error(self):
        """Test that missing 'position' key raises ValueError."""
        with pytest.raises(ValueError, match="missing required keys.*position"):
            DroneEngageMultiTargetV0(
                targets_config=[
                    {'class_type': 'A', 'zone_id': 'z1'},
                ]
            )

    def test_missing_class_type_key_raises_error(self):
        """Test that missing 'class_type' key raises ValueError."""
        with pytest.raises(ValueError, match="missing required keys.*class_type"):
            DroneEngageMultiTargetV0(
                targets_config=[
                    {'position': (100, 200), 'zone_id': 'z1'},
                ]
            )

    def test_missing_zone_id_key_raises_error(self):
        """Test that missing 'zone_id' key raises ValueError."""
        with pytest.raises(ValueError, match="missing required keys.*zone_id"):
            DroneEngageMultiTargetV0(
                targets_config=[
                    {'position': (100, 200), 'class_type': 'A'},
                ]
            )

    def test_non_dict_item_raises_error(self):
        """Test that non-dict item in targets_config raises ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            DroneEngageMultiTargetV0(
                targets_config=[
                    {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
                    "not a dict",
                ]
            )

    def test_environment_instantiation_defaults(self):
        """Test that environment uses default parameters correctly."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        assert env.world_size == (1000.0, 1000.0)
        assert env.max_steps == 100
        assert env.scenario_id == "default"
        assert env.drone_position == (100.0, 100.0)
        assert env.drone_ammo_max == 10
        assert env.drone_damage_per_shot == 30.0

    def test_environment_instantiation_custom(self):
        """Test that environment accepts custom parameters."""
        env = DroneEngageMultiTargetV0(
            world_size=(2000.0, 2000.0),
            max_steps=200,
            drone_position=(200.0, 200.0),
            drone_ammo_max=20,
            drone_damage_per_shot=50.0,
            targets_config=[
                {'position': (500, 500), 'class_type': 'B', 'zone_id': 'zone_custom'},
            ],
            scenario_id="custom_test",
        )
        
        assert env.world_size == (2000.0, 2000.0)
        assert env.max_steps == 200
        assert env.drone_position == (200.0, 200.0)
        assert env.drone_ammo_max == 20
        assert env.drone_damage_per_shot == 50.0
        assert env.scenario_id == "custom_test"


class TestActionSpaceConfiguration:
    """Test action space configuration."""

    def test_action_space_with_one_target(self):
        """Test action space sizing with 1 target."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 2  # 0=Idle, 1=Fire at target 0

    def test_action_space_with_two_targets(self):
        """Test action space sizing with 2 targets."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (300, 400), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        assert env.action_space.n == 3  # 0=Idle, 1-2=Fire at targets

    def test_action_space_with_twenty_targets(self):
        """Test action space sizing with 20 targets."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (i*50, i*50), 'class_type': 'A', 'zone_id': f'z{i}'}
                for i in range(20)
            ]
        )
        
        assert env.action_space.n == 21  # 0=Idle, 1-20=Fire at targets

    def test_action_space_sample(self):
        """Test that action space can generate samples."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (300, 400), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        sample = env.action_space.sample()
        assert 0 <= sample <= 2

    def test_action_space_contains(self):
        """Test action space contains validation."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        assert env.action_space.contains(0)
        assert env.action_space.contains(1)
        assert not env.action_space.contains(2)
        assert not env.action_space.contains(-1)


class TestObservationSpaceConfiguration:
    """Test observation space configuration."""

    def test_observation_space_with_one_target(self):
        """Test observation space sizing with 1 target."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (7,)  # 4 + 1*3
        assert env.observation_space.dtype == np.float32

    def test_observation_space_with_five_targets(self):
        """Test observation space sizing with 5 targets."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (i*100, i*100), 'class_type': 'A', 'zone_id': f'z{i}'}
                for i in range(5)
            ]
        )
        
        assert env.observation_space.shape == (19,)  # 4 + 5*3

    def test_observation_space_with_twenty_targets(self):
        """Test observation space sizing with 20 targets."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (i*50, i*50), 'class_type': 'A', 'zone_id': f'z{i}'}
                for i in range(20)
            ]
        )
        
        assert env.observation_space.shape == (64,)  # 4 + 20*3

    def test_observation_space_bounds(self):
        """Test observation space bounds."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        assert env.observation_space.low[0] == 0.0
        assert np.isinf(env.observation_space.high[0])

    def test_state_initialization_pending(self):
        """Test that state objects are None before reset."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        assert env.drone is None
        assert env.targets is None
        assert env.world is None


class TestResetLogic:
    """Test reset functionality."""

    def test_reset_returns_valid_tuple(self):
        """Test that reset returns (observation, info) tuple."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_reset_observation_shape_and_type(self):
        """Test that reset observation has correct shape and dtype."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (300, 400), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        obs, _ = env.reset()
        
        assert obs.shape == (10,)  # 4 + 2*3
        assert obs.dtype == np.float32

    def test_reset_initializes_targets(self):
        """Test that reset initializes all targets from config."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (300, 400), 'class_type': 'B', 'zone_id': 'z2'},
                {'position': (500, 600), 'class_type': 'C', 'zone_id': 'z3'},
            ]
        )
        
        env.reset()
        
        assert len(env.targets) == 3
        assert env.targets[0].class_type == 'A'
        assert env.targets[1].class_type == 'B'
        assert env.targets[2].class_type == 'C'

    def test_reset_initializes_drone_state(self):
        """Test that reset initializes drone with max ammo."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=15,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        
        assert env.drone is not None
        assert env.drone.ammo == 15
        assert env.drone.ammo_max == 15

    def test_reset_initializes_world_state(self):
        """Test that reset initializes world with time_step=0."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        
        assert env.world is not None
        assert env.world.time_step == 0

    def test_reset_with_seed(self):
        """Test that reset accepts seed parameter."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        assert np.array_equal(obs1, obs2)

    def test_reset_info_dict(self):
        """Test that reset returns info dict with required keys."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        _, info = env.reset()
        
        assert 'step_index' in info
        assert 'scenario_id' in info
        assert info['step_index'] == 0

    def test_reset_multiple_times(self):
        """Test that multiple resets properly reinitialize state."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=10,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        env.step(1)  # Fire to modify state
        
        assert env.drone.ammo == 9
        
        env.reset()
        
        assert env.drone.ammo == 10  # Should be reset
        assert env.world.time_step == 0


class TestObservationComputation:
    """Test observation vector computation."""

    def test_initial_observation_values(self):
        """Test that initial observation has expected values."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=10,
            max_steps=100,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        obs, _ = env.reset()
        
        # Drone features
        assert obs[0] == 1.0  # ammo_norm
        assert obs[1] == 0.0  # time_progress
        assert obs[2] == 0.0  # reserved
        assert obs[3] == 0.0  # reserved
        
        # Target features
        assert obs[4] == 1.0  # target HP norm
        assert obs[5] > 0     # distance (positive)
        assert obs[6] == 1.0  # active

    def test_observation_updates_after_step(self):
        """Test that observation reflects state changes."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=50.0,
            max_steps=100,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        obs1, _ = env.reset()
        obs2, _, _, _, _ = env.step(1)  # Fire at target
        
        # Ammo should decrease
        assert obs2[0] < obs1[0]
        
        # Time progress should increase
        assert obs2[1] > obs1[1]
        
        # Target HP should decrease
        assert obs2[4] < obs1[4]

    def test_hp_normalization_different_classes(self):
        """Test HP normalization for different target classes."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=50.0,
            targets_config=[
                {'position': (100, 100), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
                {'position': (200, 200), 'class_type': 'B', 'zone_id': 'z2'},  # 150 HP
                {'position': (300, 300), 'class_type': 'C', 'zone_id': 'z3'},  # 200 HP
            ]
        )
        
        env.reset()
        
        # Damage each target once
        env.step(1)  # Target 0: 100 - 50 = 50 HP
        env.step(2)  # Target 1: 150 - 50 = 100 HP
        obs, _, _, _, _ = env.step(3)  # Target 2: 200 - 50 = 150 HP
        
        # Check normalized HP values
        assert abs(obs[4] - 0.5) < 0.01        # 50/100 = 0.5
        assert abs(obs[7] - (100/150)) < 0.01  # 100/150 ≈ 0.667
        assert abs(obs[10] - 0.75) < 0.01      # 150/200 = 0.75

    def test_distance_computation(self):
        """Test that distance is computed correctly."""
        env = DroneEngageMultiTargetV0(
            drone_position=(0.0, 0.0),
            targets_config=[
                {'position': (300, 400), 'class_type': 'A', 'zone_id': 'z1'},  # Distance = 500
            ]
        )
        
        obs, _ = env.reset()
        
        expected_distance = np.sqrt(300**2 + 400**2)
        assert abs(obs[5] - expected_distance) < 0.01

    def test_active_status_as_float(self):
        """Test that active status is converted to float."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=100.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        obs1, _ = env.reset()
        assert obs1[6] == 1.0  # Active
        
        obs2, _, _, _, _ = env.step(1)  # Neutralize target
        assert obs2[6] == 0.0  # Inactive


class TestActionValidation:
    """Test action validation and bounds checking."""

    def test_valid_idle_action(self):
        """Test that action 0 (idle) is valid."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        # Should not raise error
        env.step(0)

    def test_valid_fire_actions(self):
        """Test that fire actions within range are valid."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (300, 400), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        env.reset()
        env.step(1)  # Fire at target 0
        env.step(2)  # Fire at target 1

    def test_negative_action_raises_error(self):
        """Test that negative action raises ValueError."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        with pytest.raises(ValueError, match="Invalid action.*-1"):
            env.step(-1)

    def test_action_out_of_range_raises_error(self):
        """Test that action > num_targets raises ValueError."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        with pytest.raises(ValueError, match="Invalid action.*2"):
            env.step(2)

    def test_error_message_includes_valid_range(self):
        """Test that error message shows valid action range."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (300, 400), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        env.reset()
        with pytest.raises(ValueError, match="1-2"):
            env.step(3)


class TestFireMechanics:
    """Test fire action mechanics and target damage."""

    def test_fire_decrements_ammo(self):
        """Test that firing decrements ammo."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=10,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        assert env.drone.ammo == 10
        
        env.step(1)  # Fire
        assert env.drone.ammo == 9

    def test_fire_reduces_target_hp(self):
        """Test that firing reduces target HP."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=30.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        assert env.targets[0].hp_current == 100.0
        
        env.step(1)  # Fire
        assert env.targets[0].hp_current == 70.0

    def test_target_neutralization(self):
        """Test that target is neutralized when HP reaches zero."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=100.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        assert env.targets[0].is_active is True
        
        env.step(1)  # Fire (neutralize)
        
        assert env.targets[0].hp_current == 0.0
        assert env.targets[0].is_active is False

    def test_fire_at_inactive_target_spends_ammo(self):
        """Test that firing at inactive target spends ammo but has no effect."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=100.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        env.step(1)  # Neutralize target
        
        assert env.targets[0].is_active is False
        ammo_before = env.drone.ammo
        
        env.step(1)  # Fire at inactive target
        
        assert env.drone.ammo == ammo_before - 1  # Ammo spent
        assert env.targets[0].hp_current == 0.0    # HP unchanged

    def test_fire_with_no_ammo_has_no_effect(self):
        """Test that firing with no ammo has no effect."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=1,
            drone_damage_per_shot=30.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        env.step(1)  # Use last ammo
        
        assert env.drone.ammo == 0
        hp_before = env.targets[0].hp_current
        
        env.step(1)  # Try to fire with no ammo
        
        assert env.drone.ammo == 0
        assert env.targets[0].hp_current == hp_before

    def test_independent_target_damage(self):
        """Test that targets are damaged independently."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=50.0,
            targets_config=[
                {'position': (100, 100), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (200, 200), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        env.reset()
        
        env.step(1)  # Fire at target 0
        assert env.targets[0].hp_current == 50.0
        assert env.targets[1].hp_current == 150.0  # Unchanged
        
        env.step(2)  # Fire at target 1
        assert env.targets[0].hp_current == 50.0   # Unchanged
        assert env.targets[1].hp_current == 100.0


class TestRewardComputation:
    """Test reward computation logic."""

    def test_idle_reward_is_zero(self):
        """Test that idle action returns zero reward."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        _, reward, _, _, _ = env.step(0)
        
        assert reward == 0.0

    def test_fire_without_neutralization_zero_reward(self):
        """Test that firing without neutralization returns zero reward."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=30.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        _, reward, _, _, _ = env.step(1)
        
        assert reward == 0.0

    def test_neutralization_reward_is_one(self):
        """Test that neutralizing a target returns reward of 1.0."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=100.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        _, reward, _, _, _ = env.step(1)
        
        assert reward == 1.0

    def test_multiple_neutralizations_accumulate(self):
        """Test that multiple neutralizations accumulate rewards."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=100.0,
            targets_config=[
                {'position': (100, 100), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (200, 200), 'class_type': 'A', 'zone_id': 'z2'},
            ]
        )
        
        env.reset()
        
        _, r1, _, _, _ = env.step(1)  # Neutralize target 0
        _, r2, _, _, _ = env.step(2)  # Neutralize target 1
        
        assert r1 == 1.0
        assert r2 == 1.0

    def test_fire_at_inactive_zero_reward(self):
        """Test that firing at inactive target returns zero reward."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=100.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        env.step(1)  # Neutralize
        
        _, reward, _, _, _ = env.step(1)  # Fire at inactive
        
        assert reward == 0.0


class TestTerminationLogic:
    """Test episode termination conditions."""

    def test_all_targets_neutralized_terminates(self):
        """Test that episode terminates when all targets neutralized."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=100.0,
            targets_config=[
                {'position': (100, 100), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (200, 200), 'class_type': 'A', 'zone_id': 'z2'},
            ]
        )
        
        env.reset()
        
        _, _, terminated, _, _ = env.step(1)
        assert terminated is False
        
        _, _, terminated, _, info = env.step(2)
        assert terminated is True
        assert info['done_reason'] == 'all_targets_neutralized'

    def test_no_ammo_terminates(self):
        """Test that episode terminates when ammo exhausted."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=1,
            drone_damage_per_shot=30.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        _, _, terminated, _, info = env.step(1)
        
        assert terminated is True
        assert info['done_reason'] == 'no_ammo'

    def test_max_steps_truncates(self):
        """Test that episode truncates at max_steps."""
        env = DroneEngageMultiTargetV0(
            max_steps=3,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        
        env.step(0)  # Step 1
        env.step(0)  # Step 2
        _, _, _, truncated, info = env.step(0)  # Step 3
        
        assert truncated is True
        assert info['done_reason'] == 'max_steps'

    def test_partial_neutralization_not_terminated(self):
        """Test that partial neutralization doesn't terminate episode."""
        env = DroneEngageMultiTargetV0(
            drone_damage_per_shot=100.0,
            targets_config=[
                {'position': (100, 100), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (200, 200), 'class_type': 'A', 'zone_id': 'z2'},
            ]
        )
        
        env.reset()
        _, _, terminated, _, _ = env.step(1)  # Neutralize only target 0
        
        assert terminated is False


class TestInfoDictionary:
    """Test info dictionary contents."""

    def test_info_contains_required_keys(self):
        """Test that info dict contains all required keys."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        _, _, _, _, info = env.step(0)
        
        assert 'step_index' in info
        assert 'scenario_id' in info
        assert 'ammo' in info
        assert 'target_hps' in info
        assert 'target_active' in info
        assert 'target_classes' in info
        assert 'target_zones' in info

    def test_info_arrays_correct_length(self):
        """Test that info arrays have correct length."""
        env = DroneEngageMultiTargetV0(
            targets_config=[
                {'position': (100, 100), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (200, 200), 'class_type': 'B', 'zone_id': 'z2'},
                {'position': (300, 300), 'class_type': 'C', 'zone_id': 'z3'},
            ]
        )
        
        env.reset()
        _, _, _, _, info = env.step(0)
        
        assert len(info['target_hps']) == 3
        assert len(info['target_active']) == 3
        assert len(info['target_classes']) == 3
        assert len(info['target_zones']) == 3

    def test_info_values_reflect_state(self):
        """Test that info values reflect current state."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=50.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'zone_1'},
            ]
        )
        
        env.reset()
        _, _, _, _, info = env.step(1)
        
        assert info['ammo'] == 9
        assert info['target_hps'][0] == 50.0
        assert info['target_active'][0] is True
        assert info['target_classes'][0] == 'A'
        assert info['target_zones'][0] == 'zone_1'


class TestEpisodeIntegration:
    """Test full episode scenarios."""

    def test_full_episode_all_targets_neutralized(self):
        """Test complete episode with all targets neutralized."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=50.0,
            targets_config=[
                {'position': (100, 100), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (200, 200), 'class_type': 'A', 'zone_id': 'z2'},
            ]
        )
        
        env.reset()
        
        total_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 10:
            # Simple policy: fire at first active target
            action = 0
            for i, target in enumerate(env.targets):
                if target.is_active:
                    action = i + 1
                    break
            
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        assert done is True
        assert total_reward == 2.0
        assert info['done_reason'] == 'all_targets_neutralized'

    def test_full_episode_partial_success(self):
        """Test episode ending with partial success (no ammo)."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=3,
            drone_damage_per_shot=50.0,
            targets_config=[
                {'position': (100, 100), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
                {'position': (200, 200), 'class_type': 'B', 'zone_id': 'z2'},  # 150 HP
            ]
        )
        
        env.reset()
        
        # Neutralize target 0 (2 shots)
        env.step(1)
        env.step(1)
        
        # Fire at target 1 once (1 shot, ammo exhausted)
        _, _, terminated, _, info = env.step(2)
        
        assert terminated is True
        assert info['done_reason'] == 'no_ammo'
        assert env.targets[0].is_active is False  # First target neutralized
        assert env.targets[1].is_active is True   # Second target still active

    def test_determinism_with_seed(self):
        """Test that episodes are deterministic with same seed."""
        env = DroneEngageMultiTargetV0(
            drone_ammo_max=10,
            drone_damage_per_shot=50.0,
            targets_config=[
                {'position': (100, 200), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        # Episode 1
        env.reset(seed=42)
        env.step(1)
        obs1, _, _, _, _ = env.step(1)
        
        # Episode 2
        env.reset(seed=42)
        env.step(1)
        obs2, _, _, _, _ = env.step(1)
        
        assert np.array_equal(obs1, obs2)
