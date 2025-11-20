"""
Tests for DroneEngageZKMRTA-v0 environment.
"""

import pytest
import numpy as np
from pettingzoo.test import parallel_api_test

from tabula_drone.envs.drone_engage_zk_mrta_v0 import (
    DroneEngageZKMRTA,
    DroneStateZK,
)
from tabula_drone.core.states import (
    TargetState,
    WorldState,
    DEFAULT_CLASS_HP_MAPPING,
)


class TestConstructorValidation:
    """Test constructor parameter validation."""

    def test_valid_configuration(self):
        """Test that environment can be instantiated with valid configuration."""
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        assert env.num_drones == 2
        assert env.num_targets == 1
        assert len(env.possible_agents) == 2

    def test_empty_drones_config_raises_error(self):
        """Test that empty drones_config raises ValueError."""
        with pytest.raises(ValueError, match="at least one drone"):
            DroneEngageZKMRTA(
                drones_config=[],
                targets_config=[
                    {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                ]
            )

    def test_none_drones_config_raises_error(self):
        """Test that None drones_config raises ValueError."""
        with pytest.raises(ValueError, match="at least one drone"):
            DroneEngageZKMRTA(
                drones_config=None,
                targets_config=[
                    {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                ]
            )

    def test_empty_targets_config_raises_error(self):
        """Test that empty targets_config raises ValueError."""
        with pytest.raises(ValueError, match="at least one target"):
            DroneEngageZKMRTA(
                drones_config=[{'position': (100.0, 100.0)}],
                targets_config=[]
            )

    def test_none_targets_config_raises_error(self):
        """Test that None targets_config raises ValueError."""
        with pytest.raises(ValueError, match="at least one target"):
            DroneEngageZKMRTA(
                drones_config=[{'position': (100.0, 100.0)}],
                targets_config=None
            )

    def test_missing_position_in_drone_raises_error(self):
        """Test that missing 'position' in drone config raises ValueError."""
        with pytest.raises(ValueError, match="missing required key.*position"):
            DroneEngageZKMRTA(
                drones_config=[{}],
                targets_config=[
                    {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                ]
            )

    def test_missing_position_in_target_raises_error(self):
        """Test that missing 'position' in target config raises ValueError."""
        with pytest.raises(ValueError, match="missing required keys.*position"):
            DroneEngageZKMRTA(
                drones_config=[{'position': (100.0, 100.0)}],
                targets_config=[
                    {'class_type': 'A', 'zone_id': 'z1'},
                ]
            )

    def test_missing_class_type_in_target_raises_error(self):
        """Test that missing 'class_type' in target config raises ValueError."""
        with pytest.raises(ValueError, match="missing required keys.*class_type"):
            DroneEngageZKMRTA(
                drones_config=[{'position': (100.0, 100.0)}],
                targets_config=[
                    {'position': (500.0, 500.0), 'zone_id': 'z1'},
                ]
            )

    def test_missing_zone_id_in_target_raises_error(self):
        """Test that missing 'zone_id' in target config raises ValueError."""
        with pytest.raises(ValueError, match="missing required keys.*zone_id"):
            DroneEngageZKMRTA(
                drones_config=[{'position': (100.0, 100.0)}],
                targets_config=[
                    {'position': (500.0, 500.0), 'class_type': 'A'},
                ]
            )

    def test_non_dict_drone_raises_error(self):
        """Test that non-dict item in drones_config raises ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            DroneEngageZKMRTA(
                drones_config=[
                    {'position': (100.0, 100.0)},
                    "not a dict"
                ],
                targets_config=[
                    {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                ]
            )

    def test_non_dict_target_raises_error(self):
        """Test that non-dict item in targets_config raises ValueError."""
        with pytest.raises(ValueError, match="must be a dict"):
            DroneEngageZKMRTA(
                drones_config=[{'position': (100.0, 100.0)}],
                targets_config=[
                    {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                    "not a dict"
                ]
            )

    def test_environment_instantiation_defaults(self):
        """Test that environment uses default parameters correctly."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        assert env.world_size == (1000.0, 1000.0)
        assert env.max_steps == 100
        assert env.scenario_id == "zk_mrta_baseline"
        assert env.drone_damage_per_shot == 30.0

    def test_environment_instantiation_custom(self):
        """Test that environment accepts custom parameters."""
        env = DroneEngageZKMRTA(
            world_size=(2000.0, 2000.0),
            max_steps=200,
            drone_damage_per_shot=50.0,
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'B', 'zone_id': 'custom'},
            ],
            scenario_id="custom_test",
        )
        
        assert env.world_size == (2000.0, 2000.0)
        assert env.max_steps == 200
        assert env.drone_damage_per_shot == 50.0
        assert env.scenario_id == "custom_test"

    def test_agent_ids_auto_generated(self):
        """Test that agent IDs are auto-generated as drone_0, drone_1, etc."""
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
                {'position': (300.0, 300.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        assert env.possible_agents == ['drone_0', 'drone_1', 'drone_2']
        assert env.agents == ['drone_0', 'drone_1', 'drone_2']


class TestResetLogic:
    """Test reset functionality."""

    def test_reset_returns_valid_tuple(self):
        """Test that reset returns (observations, infos) tuple."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_observation_dict_keys(self):
        """Test that reset observations dict has all agent IDs as keys."""
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        obs, _ = env.reset()
        
        assert set(obs.keys()) == {'drone_0', 'drone_1'}
        assert all(isinstance(v, np.ndarray) for v in obs.values())

    def test_reset_observation_shape(self):
        """Test that reset observations have correct shape."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (600.0, 600.0), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        obs, _ = env.reset()
        
        # Shape should be (3 * num_targets,)
        assert obs['drone_0'].shape == (6,)
        assert obs['drone_0'].dtype == np.float32

    def test_reset_initializes_drones(self):
        """Test that reset initializes all drones from config."""
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
                {'position': (300.0, 300.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        
        assert len(env.drones) == 3
        assert env.drones[0].id == 'drone_0'
        assert env.drones[1].id == 'drone_1'
        assert env.drones[2].id == 'drone_2'

    def test_reset_initializes_drone_ammo_used(self):
        """Test that reset initializes drones with ammo_used=0."""
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        
        assert env.drones[0].ammo_used == 0
        assert env.drones[1].ammo_used == 0

    def test_reset_initializes_targets(self):
        """Test that reset initializes all targets from config."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (600.0, 600.0), 'class_type': 'B', 'zone_id': 'z2'},
                {'position': (700.0, 700.0), 'class_type': 'C', 'zone_id': 'z3'},
            ]
        )
        
        env.reset()
        
        assert len(env.targets) == 3
        assert env.targets[0].class_type == 'A'
        assert env.targets[1].class_type == 'B'
        assert env.targets[2].class_type == 'C'

    def test_reset_initializes_target_hp(self):
        """Test that reset initializes targets with correct HP from class mapping."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (600.0, 600.0), 'class_type': 'B', 'zone_id': 'z2'},
                {'position': (700.0, 700.0), 'class_type': 'C', 'zone_id': 'z3'},
            ]
        )
        
        env.reset()
        
        assert env.targets[0].hp_current == 100.0
        assert env.targets[1].hp_current == 150.0
        assert env.targets[2].hp_current == 200.0
        assert all(t.is_active for t in env.targets)

    def test_reset_initializes_world_state(self):
        """Test that reset initializes world with time_step=0."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        
        assert env.world is not None
        assert env.world.time_step == 0

    def test_reset_with_seed(self):
        """Test that reset accepts seed parameter."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        assert np.array_equal(obs1['drone_0'], obs2['drone_0'])

    def test_reset_info_dict(self):
        """Test that reset returns info dict with required keys."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        _, info = env.reset()
        
        assert 'step_index' in info
        assert 'scenario_id' in info
        assert 'ammo_used' in info
        assert 'target_hps' in info
        assert 'target_active' in info
        assert info['step_index'] == 0

    def test_reset_multiple_times(self):
        """Test that multiple resets properly reinitialize state."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        actions = {'drone_0': 1}
        env.step(actions)  # Fire to modify state
        
        assert env.drones[0].ammo_used == 1
        
        env.reset()
        
        assert env.drones[0].ammo_used == 0  # Should be reset
        assert env.world.time_step == 0


class TestZKObservationCompliance:
    """Test Zero-Knowledge observation compliance (CRITICAL)."""

    def test_observations_contain_target_positions(self):
        """Test that observations contain target x, y positions."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 600.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        obs, _ = env.reset()
        
        # Observation format: [x, y, active] per target
        assert obs['drone_0'][0] == 500.0  # x position
        assert obs['drone_0'][1] == 600.0  # y position

    def test_observations_contain_binary_active_status(self):
        """Test that observations contain binary active status (1.0 or 0.0)."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=100.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 600.0), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (700.0, 800.0), 'class_type': 'A', 'zone_id': 'z2'},
            ]
        )
        
        obs, _ = env.reset()
        
        # Both targets active initially
        assert obs['drone_0'][2] == 1.0  # target 0 active
        assert obs['drone_0'][5] == 1.0  # target 1 active
        
        # Neutralize target 0
        actions = {'drone_0': 1}
        obs, _, _, _, _ = env.step(actions)
        
        # Now target 0 inactive, target 1 still active
        assert obs['drone_0'][2] == 0.0  # target 0 inactive
        assert obs['drone_0'][5] == 1.0  # target 1 active

    def test_observations_do_not_contain_hp_values(self):
        """Test that observations do NOT contain HP values (ZK constraint)."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=30.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 600.0), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        obs, _ = env.reset()
        
        # Fire to reduce HP (100 - 30 = 70 HP)
        actions = {'drone_0': 1}
        obs, _, _, _, info = env.step(actions)
        
        # HP is 70 but not in observation
        assert info['target_hps'][0] == 70.0
        
        # Observation only contains: [x=500, y=600, active=1]
        assert obs['drone_0'][0] == 500.0
        assert obs['drone_0'][1] == 600.0
        assert obs['drone_0'][2] == 1.0
        
        # No other values that could be HP
        assert len(obs['drone_0']) == 3  # Only these 3 values

    def test_observations_do_not_contain_class_types(self):
        """Test that observations do NOT contain class types (ZK constraint)."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 600.0), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (700.0, 800.0), 'class_type': 'B', 'zone_id': 'z2'},
                {'position': (900.0, 900.0), 'class_type': 'C', 'zone_id': 'z3'},
            ]
        )
        
        obs, info = env.reset()
        
        # Info contains class types (for logging)
        assert info['target_classes'] == ['A', 'B', 'C']
        
        # Observation only contains positions and active status
        # No way to distinguish class types from observation
        assert obs['drone_0'].shape == (9,)  # 3 targets * 3 values
        
        # All values are floats (positions and binary active)
        assert obs['drone_0'].dtype == np.float32

    def test_observations_do_not_contain_ammo_usage(self):
        """Test that observations do NOT contain ammo usage (ZK constraint)."""
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 600.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        obs1, _ = env.reset()
        
        # Fire with drone_0
        actions = {'drone_0': 1, 'drone_1': 0}
        obs2, _, _, _, info = env.step(actions)
        
        # Info contains ammo usage
        assert info['ammo_used']['drone_0'] == 1
        assert info['ammo_used']['drone_1'] == 0
        
        # Observations remain identical (no ammo info)
        assert np.array_equal(obs2['drone_0'], obs2['drone_1'])

    def test_all_agents_receive_identical_observations(self):
        """Test that all agents receive identical observations (global visibility)."""
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
                {'position': (300.0, 300.0)},
            ],
            targets_config=[
                {'position': (500.0, 600.0), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (700.0, 800.0), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        obs, _ = env.reset()
        
        # All drones see the same thing
        assert np.array_equal(obs['drone_0'], obs['drone_1'])
        assert np.array_equal(obs['drone_1'], obs['drone_2'])
        
        # After actions
        actions = {'drone_0': 1, 'drone_1': 2, 'drone_2': 0}
        obs, _, _, _, _ = env.step(actions)
        
        # Still identical
        assert np.array_equal(obs['drone_0'], obs['drone_1'])
        assert np.array_equal(obs['drone_1'], obs['drone_2'])

    def test_observation_values_are_floats_only(self):
        """Test that observation values are all floats (no encoded information)."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 600.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        obs, _ = env.reset()
        
        # All values are floats
        assert obs['drone_0'].dtype == np.float32
        
        # All values are reasonable floats (positions or 0/1)
        for val in obs['drone_0']:
            assert isinstance(float(val), float)
            # Either a position (>= 0) or binary (0 or 1)
            assert val >= 0.0


class TestActionValidation:
    """Test action validation."""

    def test_valid_noop_action(self):
        """Test that action 0 (NoOp) is valid for all agents."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        # Should not raise error
        actions = {'drone_0': 0}
        env.step(actions)

    def test_valid_fire_actions(self):
        """Test that fire actions within range are valid."""
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (600.0, 600.0), 'class_type': 'B', 'zone_id': 'z2'},
            ]
        )
        
        env.reset()
        actions = {'drone_0': 1, 'drone_1': 2}
        env.step(actions)

    def test_missing_agent_raises_error(self):
        """Test that missing agent in actions raises ValueError."""
        env = DroneEngageZKMRTA(
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        with pytest.raises(ValueError, match="Actions must be provided for all agents"):
            env.step({'drone_0': 1})  # Missing drone_1

    def test_action_out_of_range_raises_error(self):
        """Test that action > num_targets raises ValueError."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        with pytest.raises(ValueError, match="must be in range"):
            env.step({'drone_0': 99})

    def test_negative_action_raises_error(self):
        """Test that negative action raises ValueError."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        with pytest.raises(ValueError, match="must be in range"):
            env.step({'drone_0': -1})

    def test_non_integer_action_raises_error(self):
        """Test that non-integer action raises ValueError."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        with pytest.raises(ValueError, match="must be an integer"):
            env.step({'drone_0': 'fire'})


class TestDamageMechanics:
    """Test damage application and aggregation."""

    def test_single_drone_damage_application(self):
        """Test that single drone damage is applied correctly."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=30.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        assert env.targets[0].hp_current == 100.0
        
        actions = {'drone_0': 1}
        env.step(actions)
        
        assert env.targets[0].hp_current == 70.0

    def test_multiple_drones_damage_aggregation(self):
        """Test that damage from multiple drones aggregates correctly."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=30.0,
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
                {'position': (300.0, 300.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        
        # All 3 drones fire at same target (3 * 30 = 90 damage)
        actions = {'drone_0': 1, 'drone_1': 1, 'drone_2': 1}
        env.step(actions)
        
        assert env.targets[0].hp_current == 10.0

    def test_target_neutralization(self):
        """Test that target is neutralized when HP reaches zero."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=100.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        assert env.targets[0].is_active is True
        
        actions = {'drone_0': 1}
        env.step(actions)
        
        assert env.targets[0].hp_current == 0.0
        assert env.targets[0].is_active is False

    def test_hp_clamped_at_zero(self):
        """Test that HP is clamped at 0 (no negative HP)."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=150.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        
        actions = {'drone_0': 1}
        env.step(actions)
        
        assert env.targets[0].hp_current == 0.0  # Not negative

    def test_fire_at_inactive_target_has_no_effect(self):
        """Test that firing at inactive target has no effect on HP."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=100.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        
        # Neutralize target
        actions = {'drone_0': 1}
        env.step(actions)
        assert env.targets[0].is_active is False
        
        # Fire at inactive target
        env.step(actions)
        
        assert env.targets[0].hp_current == 0.0  # Still 0


class TestRewardComputation:
    """Test shared cooperative reward computation."""

    def test_noop_reward_is_zero(self):
        """Test that NoOp action returns zero reward."""
        env = DroneEngageZKMRTA(
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        actions = {'drone_0': 0}
        _, rewards, _, _, _ = env.step(actions)
        
        assert rewards['drone_0'] == 0.0

    def test_fire_without_neutralization_zero_reward(self):
        """Test that firing without neutralization returns zero reward."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=30.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        actions = {'drone_0': 1}
        _, rewards, _, _, _ = env.step(actions)
        
        assert rewards['drone_0'] == 0.0

    def test_single_drone_neutralization_reward(self):
        """Test that single drone neutralizing target gets +1.0 reward."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=100.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        actions = {'drone_0': 1}
        _, rewards, _, _, _ = env.step(actions)
        
        assert rewards['drone_0'] == 1.0

    def test_multiple_drones_shared_reward(self):
        """Test that all drones firing at neutralized target get +1.0 reward."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=50.0,
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        
        # Both drones fire at target (2 * 50 = 100 damage, neutralized)
        actions = {'drone_0': 1, 'drone_1': 1}
        _, rewards, _, _, _ = env.step(actions)
        
        assert rewards['drone_0'] == 1.0
        assert rewards['drone_1'] == 1.0

    def test_fire_at_inactive_zero_reward(self):
        """Test that firing at inactive target returns zero reward."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=100.0,
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        
        # Drone 0 neutralizes
        actions = {'drone_0': 1, 'drone_1': 0}
        _, rewards1, _, _, _ = env.step(actions)
        assert rewards1['drone_0'] == 1.0
        
        # Drone 1 fires at inactive target
        actions = {'drone_0': 0, 'drone_1': 1}
        _, rewards2, _, _, _ = env.step(actions)
        assert rewards2['drone_1'] == 0.0


class TestTerminationLogic:
    """Test episode termination conditions."""

    def test_all_targets_neutralized_terminates(self):
        """Test that episode terminates when all targets neutralized."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=100.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
            ]
        )
        
        env.reset()
        
        actions = {'drone_0': 1}
        _, _, terminations, _, info = env.step(actions)
        
        assert terminations['drone_0'] is True
        assert info['done_reason'] == 'all_targets_neutralized'

    def test_max_steps_truncates(self):
        """Test that episode truncates at max_steps."""
        env = DroneEngageZKMRTA(
            max_steps=3,
            drone_damage_per_shot=1.0,  # Weak damage
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
            ]
        )
        
        env.reset()
        
        actions = {'drone_0': 1}
        env.step(actions)  # Step 1
        env.step(actions)  # Step 2
        _, _, _, truncations, info = env.step(actions)  # Step 3
        
        assert truncations['drone_0'] is True
        assert info['done_reason'] == 'max_steps'

    def test_partial_neutralization_not_terminated(self):
        """Test that partial neutralization doesn't terminate episode."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=100.0,
            drones_config=[{'position': (100.0, 100.0)}],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},
                {'position': (600.0, 600.0), 'class_type': 'A', 'zone_id': 'z2'},
            ]
        )
        
        env.reset()
        
        # Neutralize only first target
        actions = {'drone_0': 1}
        _, _, terminations, truncations, _ = env.step(actions)
        
        assert terminations['drone_0'] is False
        assert truncations['drone_0'] is False


class TestEpisodeIntegration:
    """Test full episode scenarios."""

    def test_full_episode_all_targets_neutralized(self):
        """Test complete episode with all targets neutralized."""
        env = DroneEngageZKMRTA(
            drone_damage_per_shot=50.0,
            drones_config=[
                {'position': (100.0, 100.0)},
                {'position': (200.0, 200.0)},
            ],
            targets_config=[
                {'position': (500.0, 500.0), 'class_type': 'A', 'zone_id': 'z1'},  # 100 HP
                {'position': (600.0, 600.0), 'class_type': 'A', 'zone_id': 'z2'},  # 100 HP
            ]
        )
        
        obs, _ = env.reset()
        
        total_rewards = {agent_id: 0.0 for agent_id in env.agents}
        done = False
        steps = 0
        
        while not done and steps < 10:
            # Simple policy: both drones fire at first active target
            target_idx = 1  # Start with target 0
            for i, target in enumerate(env.targets):
                if target.is_active:
                    target_idx = i + 1
                    break
            
            actions = {'drone_0': target_idx, 'drone_1': target_idx}
            obs, rewards, terminations, truncations, info = env.step(actions)
            
            for agent_id in env.agents:
                total_rewards[agent_id] += rewards[agent_id]
            
            done = terminations[env.agents[0]] or truncations[env.agents[0]]
            steps += 1
        
        assert done is True
        assert info['done_reason'] == 'all_targets_neutralized'
        # Both drones participated in both neutralizations
        assert total_rewards['drone_0'] == 2.0
        assert total_rewards['drone_1'] == 2.0
