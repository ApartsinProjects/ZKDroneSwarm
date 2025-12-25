"""
Tests for OptimalAssignmentOracle policy.

Verifies:
- No two agents share a target (collision-free)
- Inactive targets are never selected
- Total score matches selected pairs
- Edge cases (zero active targets, shape mismatches)
"""

import pytest
import numpy as np

from tabula_drone.policies.max_damage_oracle import OptimalAssignmentOracle


class TestOptimalAssignmentOracleBasic:
    """Basic functionality tests."""

    def test_instantiation(self):
        """Test that oracle can be instantiated with valid config."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0, "shields": 5.0},
            "drone_1": {"armor": 8.0, "shields": 12.0},
        }
        oracle = OptimalAssignmentOracle(
            agent_weapon_profiles=agent_weapon_profiles,
            seed=42,
            allow_noop=True,
        )
        
        assert oracle.num_agents == 2
        assert "drone_0" in oracle.agent_ids
        assert "drone_1" in oracle.agent_ids

    def test_select_actions_returns_dict(self):
        """Test that select_actions returns Dict[str, int]."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0, "shields": 5.0},
        }
        oracle = OptimalAssignmentOracle(agent_weapon_profiles=agent_weapon_profiles)
        
        observations = {
            "drone_0": np.array([100.0, 200.0, 1.0, 300.0, 400.0, 1.0], dtype=np.float32)
        }
        targets_state = [
            {"armor": 50.0, "shields": 30.0},
            {"armor": 40.0, "shields": 20.0},
        ]
        
        actions = oracle.select_actions(observations, num_targets=2, targets_state=targets_state)
        
        assert isinstance(actions, dict)
        assert "drone_0" in actions
        assert isinstance(actions["drone_0"], int)


class TestCollisionFreeAssignment:
    """Tests ensuring no two agents share a target."""

    def test_no_collision_two_agents_two_targets(self):
        """Test that 2 agents are assigned to 2 different targets."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0, "shields": 5.0},
            "drone_1": {"armor": 8.0, "shields": 12.0},
        }
        oracle = OptimalAssignmentOracle(agent_weapon_profiles=agent_weapon_profiles)
        
        observations = {
            "drone_0": np.array([100.0, 200.0, 1.0, 300.0, 400.0, 1.0], dtype=np.float32),
            "drone_1": np.array([100.0, 200.0, 1.0, 300.0, 400.0, 1.0], dtype=np.float32),
        }
        targets_state = [
            {"armor": 50.0, "shields": 30.0},
            {"armor": 40.0, "shields": 20.0},
        ]
        
        actions = oracle.select_actions(observations, num_targets=2, targets_state=targets_state)
        
        assigned_targets = [a for a in actions.values() if a > 0]
        assert len(assigned_targets) == len(set(assigned_targets)), "Collision detected: two agents share a target"

    def test_no_collision_three_agents_four_targets(self):
        """Test 3x4 case: no collisions among assigned targets."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0, "shields": 5.0},
            "drone_1": {"armor": 8.0, "shields": 12.0},
            "drone_2": {"armor": 15.0, "shields": 3.0},
        }
        oracle = OptimalAssignmentOracle(agent_weapon_profiles=agent_weapon_profiles)
        
        obs = np.array([
            100.0, 200.0, 1.0,
            300.0, 400.0, 1.0,
            500.0, 600.0, 1.0,
            700.0, 800.0, 1.0,
        ], dtype=np.float32)
        observations = {f"drone_{i}": obs for i in range(3)}
        
        targets_state = [
            {"armor": 50.0, "shields": 30.0},
            {"armor": 40.0, "shields": 20.0},
            {"armor": 60.0, "shields": 10.0},
            {"armor": 30.0, "shields": 40.0},
        ]
        
        actions = oracle.select_actions(observations, num_targets=4, targets_state=targets_state)
        
        assigned_targets = [a for a in actions.values() if a > 0]
        assert len(assigned_targets) == 3, "All 3 agents should be assigned"
        assert len(assigned_targets) == len(set(assigned_targets)), "Collision detected"


class TestInactiveTargets:
    """Tests ensuring inactive targets are never selected."""

    def test_inactive_target_not_selected(self):
        """Test that inactive targets (active=0) are never assigned."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0, "shields": 5.0},
        }
        oracle = OptimalAssignmentOracle(agent_weapon_profiles=agent_weapon_profiles)
        
        observations = {
            "drone_0": np.array([
                100.0, 200.0, 0.0,
                300.0, 400.0, 1.0,
            ], dtype=np.float32)
        }
        targets_state = [
            {"armor": 100.0, "shields": 100.0},
            {"armor": 10.0, "shields": 10.0},
        ]
        
        actions = oracle.select_actions(observations, num_targets=2, targets_state=targets_state)
        
        assert actions["drone_0"] == 2, "Should select active target (index 1 -> action 2)"

    def test_all_inactive_returns_noop(self):
        """Test that all inactive targets results in NoOp (allow_noop=True)."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0, "shields": 5.0},
        }
        oracle = OptimalAssignmentOracle(
            agent_weapon_profiles=agent_weapon_profiles,
            allow_noop=True,
        )
        
        observations = {
            "drone_0": np.array([
                100.0, 200.0, 0.0,
                300.0, 400.0, 0.0,
            ], dtype=np.float32)
        }
        targets_state = [
            {"armor": 50.0, "shields": 30.0},
            {"armor": 40.0, "shields": 20.0},
        ]
        
        actions = oracle.select_actions(observations, num_targets=2, targets_state=targets_state)
        
        assert actions["drone_0"] == 0, "Should return NoOp when all targets inactive"

    def test_all_inactive_returns_minus_one_when_noop_disabled(self):
        """Test that all inactive targets results in -1 (allow_noop=False)."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0, "shields": 5.0},
        }
        oracle = OptimalAssignmentOracle(
            agent_weapon_profiles=agent_weapon_profiles,
            allow_noop=False,
        )
        
        observations = {
            "drone_0": np.array([
                100.0, 200.0, 0.0,
                300.0, 400.0, 0.0,
            ], dtype=np.float32)
        }
        targets_state = [
            {"armor": 50.0, "shields": 30.0},
            {"armor": 40.0, "shields": 20.0},
        ]
        
        actions = oracle.select_actions(observations, num_targets=2, targets_state=targets_state)
        
        assert actions["drone_0"] == -1, "Should return -1 when noop disabled and all inactive"


class TestScoreMatching:
    """Tests verifying total score matches selected pairs."""

    def test_score_computation_simple(self):
        """Test that assignment maximizes dot-product score."""
        agent_weapon_profiles = {
            "drone_0": {"a": 1.0, "b": 0.0},
            "drone_1": {"a": 0.0, "b": 1.0},
        }
        oracle = OptimalAssignmentOracle(agent_weapon_profiles=agent_weapon_profiles)
        
        observations = {
            "drone_0": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "drone_1": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }
        targets_state = [
            {"a": 10.0, "b": 0.0},
            {"a": 0.0, "b": 10.0},
        ]
        
        actions = oracle.select_actions(observations, num_targets=2, targets_state=targets_state)
        
        assert actions["drone_0"] == 1, "drone_0 (a=1,b=0) should target idx 0 (a=10,b=0)"
        assert actions["drone_1"] == 2, "drone_1 (a=0,b=1) should target idx 1 (a=0,b=10)"


class TestEdgeCases:
    """Edge case tests."""

    def test_more_agents_than_targets(self):
        """Test case where agents > active targets."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0},
            "drone_1": {"armor": 8.0},
            "drone_2": {"armor": 15.0},
        }
        oracle = OptimalAssignmentOracle(
            agent_weapon_profiles=agent_weapon_profiles,
            allow_noop=True,
        )
        
        observations = {
            "drone_0": np.array([100.0, 200.0, 1.0], dtype=np.float32),
            "drone_1": np.array([100.0, 200.0, 1.0], dtype=np.float32),
            "drone_2": np.array([100.0, 200.0, 1.0], dtype=np.float32),
        }
        targets_state = [{"armor": 50.0}]
        
        actions = oracle.select_actions(observations, num_targets=1, targets_state=targets_state)
        
        assigned_count = sum(1 for a in actions.values() if a > 0)
        noop_count = sum(1 for a in actions.values() if a == 0)
        
        assert assigned_count == 1, "Only one agent can be assigned to single target"
        assert noop_count == 2, "Two agents should be unassigned (NoOp)"

    def test_empty_targets_state(self):
        """Test handling of empty targets_state."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0},
        }
        oracle = OptimalAssignmentOracle(
            agent_weapon_profiles=agent_weapon_profiles,
            allow_noop=True,
        )
        
        observations = {"drone_0": np.array([], dtype=np.float32)}
        targets_state = []
        
        actions = oracle.select_actions(observations, num_targets=0, targets_state=targets_state)
        
        assert actions["drone_0"] == 0, "Should return NoOp for empty targets"

    def test_single_agent_single_target(self):
        """Test simplest case: 1 agent, 1 active target."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0, "shields": 5.0},
        }
        oracle = OptimalAssignmentOracle(agent_weapon_profiles=agent_weapon_profiles)
        
        observations = {
            "drone_0": np.array([100.0, 200.0, 1.0], dtype=np.float32)
        }
        targets_state = [{"armor": 50.0, "shields": 30.0}]
        
        actions = oracle.select_actions(observations, num_targets=1, targets_state=targets_state)
        
        assert actions["drone_0"] == 1, "Should assign to the only active target"


class TestActionIndexing:
    """Tests verifying 1-indexed action output."""

    def test_actions_are_one_indexed(self):
        """Test that returned actions are 1-indexed (not 0-indexed)."""
        agent_weapon_profiles = {
            "drone_0": {"armor": 10.0},
        }
        oracle = OptimalAssignmentOracle(agent_weapon_profiles=agent_weapon_profiles)
        
        observations = {
            "drone_0": np.array([
                100.0, 200.0, 1.0,
                300.0, 400.0, 1.0,
            ], dtype=np.float32)
        }
        targets_state = [
            {"armor": 10.0},
            {"armor": 100.0},
        ]
        
        actions = oracle.select_actions(observations, num_targets=2, targets_state=targets_state)
        
        assert actions["drone_0"] in [1, 2], "Action should be 1 or 2 (1-indexed)"
        assert actions["drone_0"] == 2, "Should prefer target with higher score (index 1 -> action 2)"
